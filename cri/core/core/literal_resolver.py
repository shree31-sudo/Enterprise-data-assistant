from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

log = logging.getLogger("LiteralResolver")

# ============================================================
# MODULE-LEVEL STATE
# ============================================================
_VALUE_CACHE:  Dict[Tuple[str, str], str]  = {}
_FK_TO_MASTER: Dict[str, Dict[str, str]]   = {}
_MASTER_INFO:  Dict[str, Dict[str, str]]   = {}

# Optional Qdrant client (set by index_literal_values)
_QDRANT_CLIENT = None
_QDRANT_COLLECTION = "literal_values"
_EMBEDDER = None
_SEARCHABLE_COLS = None


# ============================================================
# STARTUP INIT
# ============================================================
def init(meta: dict) -> None:
    """Build FK->Master and Master info maps from schema meta at startup."""
    global _FK_TO_MASTER, _MASTER_INFO
    _FK_TO_MASTER = {}
    _MASTER_INFO  = {}

    for tname, tinfo in meta.items():
        if tinfo.get("table_type", "").upper() != "MASTER":
            continue
        idents = tinfo.get("identifiers", {})
        pk = idents.get("primary_key", "")
        bk = idents.get("business_key", "")
        if not pk or not bk or pk == bk:
            continue
        _MASTER_INFO[tname] = {"pk": pk, "bk": bk}
        if pk not in _FK_TO_MASTER:
            _FK_TO_MASTER[pk] = {
                "master_table": tname,
                "master_pk":    pk,
                "business_key": bk,
            }

    log.info(
        f"[LiteralResolver] init: {len(_MASTER_INFO)} master tables, "
        f"{len(_FK_TO_MASTER)} FK mappings loaded."
    )


# ============================================================
# LITERAL EXTRACTION
# ============================================================
def extract_literal_value(question: str) -> Optional[str]:
    """
    Extract a business-key / code value from the question string.
    Returns uppercased value or None.
    Detection is purely structural — no domain keywords.
    """
    m = re.search(r"['\"]([^'\"]{2,})['\"]", question)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"\b([A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*[0-9][A-Za-z0-9]*"
        r"|[A-Za-z0-9]*[0-9][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*[A-Za-z][A-Za-z0-9]*)\b",
        question,
    )
    if m:
        val = m.group(1).upper()
        if len(val) >= 4 and re.search(r'[A-Z]', val) and re.search(r'[0-9]', val):
            log.info(f"[LiteralResolver] Extracted alphanumeric: '{val}'")
            return val

    m = re.search(r"\b([A-Za-z0-9]{2,}(?:[-_][A-Za-z0-9]{1,}){1,})\b", question)
    if m:
        val = m.group(1).upper()
        if len(val) >= 5:
            log.info(f"[LiteralResolver] Extracted separator-token: '{val}'")
            return val

    # Pure-numeric fallback — the three checks above all require at least one
    # letter, so a bare numeric record ID (e.g. "item 632128", "order 100234")
    # was previously never recognized as a literal at all and no filter hint
    # ever got built. This is a last resort, structural only (no table/column
    # keywords): a run of 5+ digits is treated as a numeric identifier. The
    # 5-digit floor is deliberate — it avoids misfiring on small numbers the
    # user commonly mentions for other reasons (e.g. "top 10", "last 30
    # days") and on plain 4-digit years, while still catching typical
    # surrogate-key-sized IDs.
    m = re.search(r"\b(\d{5,})\b", question)
    if m:
        val = m.group(1)
        log.info(f"[LiteralResolver] Extracted numeric identifier: '{val}'")
        return val

    return None


# ============================================================
# QDRANT INDEXING (Improvement 6)
# ============================================================
def index_literal_values(
    qdrant_client,
    meta: dict,
    searchable_cols: Dict[str, List[str]],
    embedder,
    collection_name: str = "literal_values",
    batch_size: int = 1000,
) -> None:
    """
    Index all distinct literal values from searchable columns into Qdrant.
    This enables fast ANN lookup for literal resolution.
    """
    global _QDRANT_CLIENT, _QDRANT_COLLECTION, _EMBEDDER, _SEARCHABLE_COLS
    _QDRANT_CLIENT = qdrant_client
    _QDRANT_COLLECTION = collection_name
    _EMBEDDER = embedder
    _SEARCHABLE_COLS = searchable_cols

    from qdrant_client.models import VectorParams, Distance, PointStruct

    try:
        # Create collection if not exists
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection_name not in collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
            )
            log.info(f"[LiteralResolver] Created Qdrant collection '{collection_name}'")
    except Exception as e:
        log.warning(f"[LiteralResolver] Qdrant collection init failed: {e} – literal index disabled")
        return

    # Collect all (table, column, value) triples
    points = []
    point_id = 0
    from config.db import get_db_connection

    with get_db_connection() as conn:
        cur = conn.cursor()
        for table, cols in searchable_cols.items():
            for col in cols:
                try:
                    cur.execute(f"SELECT DISTINCT TOP 5000 [{col}] FROM [{table}] WHERE [{col}] IS NOT NULL")
                    rows = cur.fetchall()
                    for row in rows:
                        val = row[0]
                        if val is None:
                            continue
                        val_str = str(val).strip()
                        if not val_str:
                            continue
                        vec = embedder.encode(val_str, normalize_embeddings=True).tolist()
                        points.append(PointStruct(
                            id=point_id,
                            vector=vec,
                            payload={
                                "table": table,
                                "column": col,
                                "value": val_str,
                            },
                        ))
                        point_id += 1
                        if len(points) >= batch_size:
                            qdrant_client.upsert(collection_name=collection_name, points=points)
                            points = []
                except Exception as e:
                    log.warning(f"[LiteralResolver] Failed to index {table}.{col}: {e}")
                    continue
        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)

    log.info(f"[LiteralResolver] Indexed {point_id} literal values into Qdrant")


# ============================================================
# QDRANT-BASED LITERAL LOOKUP
# ============================================================
def _literal_qdrant_search(
    literal_value: str,
    table_filter: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Tuple[str, str, float]]:
    """
    Search Qdrant for the literal value. Returns list of (table, column, score)
    sorted by descending similarity.
    """
    if _QDRANT_CLIENT is None or _EMBEDDER is None:
        return []

    try:
        vec = _EMBEDDER.encode(literal_value, normalize_embeddings=True).tolist()
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        query_filter = None
        if table_filter:
            # Use 'should' (OR) not 'must' (AND) — a point belongs to exactly one
            # table, so AND-ing multiple table conditions always returns nothing.
            # MatchAny lets Qdrant match any of the provided table names in one pass.
            query_filter = Filter(
                should=[FieldCondition(key="table", match=MatchAny(any=table_filter))]
            )
        results = _QDRANT_CLIENT.search(
            collection_name=_QDRANT_COLLECTION,
            query_vector=vec,
            limit=limit,
            query_filter=query_filter,
            score_threshold=0.55,  # lowered from 0.75 — short alphanumeric codes
            # (e.g. WOCNC0101) have poor embedding similarity; 0.75 misses them.
        )
        return [(hit.payload["table"], hit.payload["column"], hit.score) for hit in results]
    except Exception as e:
        log.warning(f"[LiteralResolver] Qdrant search failed: {e}")
        return []


# ============================================================
# SINGLE TABLE TRIAL (enhanced with Qdrant)
# ============================================================
def resolve_by_trial(
    conn,
    question:        str,   # kept for compatibility, but literal value is used
    table:           str,
    value:           str,
    max_trials:      int  = 5,
    qdrant_client        = None,
    collection_name: str  = "",
    embedder             = None,
    searchable_cols: dict = None,
    db_schema:       str  = "dbo",
) -> Optional[str]:
    """Find which column in table contains value. Returns column name or None."""
    value_upper = value.upper()
    cache_key   = (table, value_upper)
    _searchable = searchable_cols or {}

    if cache_key in _VALUE_CACHE:
        log.info(f"[LiteralResolver] Cache hit: {table}.{_VALUE_CACHE[cache_key]}")
        return _VALUE_CACHE[cache_key]

    # 1. Try Qdrant if available
    candidate_cols = []
    if qdrant_client and collection_name and embedder:
        # Use literal value itself for ANN search, not the question
        hits = _literal_qdrant_search(value_upper, table_filter=[table], limit=max_trials)
        candidate_cols = [col for _, col, _ in hits if col in _searchable.get(table, [])]

    # 2. Fallback to schema-based trial (original logic)
    if not candidate_cols:
        # Use question-based ranking (legacy)
        top_cols = _searchable.get(table, [])[:max_trials]
        candidate_cols = top_cols

    if not candidate_cols:
        return None

    cur = conn.cursor()
    for col in candidate_cols:
        try:
            cur.execute(
                f"SELECT TOP 1 1 FROM [{table}] WITH (NOLOCK) "
                f"WHERE UPPER([{col}]) = UPPER(?)",
                (value_upper,)
            )
            if cur.fetchone():
                _VALUE_CACHE[cache_key] = col
                log.info(f"[LiteralResolver] '{value_upper}' matched: {table}.{col}")
                return col
        except Exception as e:
            log.debug(f"[LiteralResolver] Trial {table}.{col}: {e}")
            continue

    return None


# ============================================================
# CROSS-TABLE RESOLUTION
# ============================================================
def resolve_literal_across_tables(
    conn,
    question:        str,
    ranked_tables:   list,
    value:           str,
    qdrant_client        = None,
    collection_name: str  = "",
    embedder             = None,
    searchable_cols: dict = None,
    db_schema:       str  = "dbo",
    meta:            dict = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Search value across ranked_tables. Returns (table, col) or (None, None).
    Pass 1 — direct scan of subgroup tables (using Qdrant if available).
    Pass 2 — probe master tables reachable via FK columns in the subgroup.
    """
    value_upper = value.upper()
    kwargs = dict(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedder=embedder,
        searchable_cols=searchable_cols,
        db_schema=db_schema,
    )

    # Pass 1: Try each table in order
    for table in ranked_tables:
        col = resolve_by_trial(conn, question, table, value_upper, **kwargs)
        if col:
            log.info(f"[LiteralResolver] '{value_upper}' -> {table}.{col}")
            return table, col

    log.info(f"[LiteralResolver] '{value_upper}' not in subgroup tables — trying master fallback.")

    # Pass 2: Master fallback via FK relationships
    if meta:
        result = resolve_via_master_tables(conn, value_upper, ranked_tables, meta)
        if result:
            master_table, bk_col = result
            log.info(f"[LiteralResolver] '{value_upper}' resolved via master: {master_table}.{bk_col}")
            return master_table, bk_col

    log.warning(f"[LiteralResolver] '{value_upper}' not found in {ranked_tables} or their masters.")
    return None, None


# ============================================================
# MASTER-TABLE FALLBACK
# ============================================================
def resolve_via_master_tables(
    conn,
    value_upper:    str,
    current_tables: list,
    meta:           dict,
) -> Optional[Tuple[str, str]]:
    """
    Search value_upper in master tables reachable via FK columns present
    in current_tables.

    Pass A — FK-scoped: only masters whose PK appears as a column in
              current_tables (original behaviour, fast and precise).
    Pass B — Prefix-broadened: if Pass A finds nothing, widen the search
              to ALL master tables whose name shares a prefix with any
              current_table (e.g. MFC_T_WORK_ORDER_OP -> MFC_M_WORK_ORDER).
              This catches the case where the transaction table links to
              a master through an intermediate table (SN_ROU -> WORK_ORDER).

    Returns (master_table, business_key_col) or None.
    """
    if not _FK_TO_MASTER:
        return None

    # ── Pass A: FK-scoped (original logic) ───────────────────────────────────
    subgroup_cols: set = set()
    for table in current_tables:
        subgroup_cols.update(meta.get(table, {}).get("columns", {}).keys())

    seen_masters_a: Dict[str, Tuple[str, str]] = {}
    if subgroup_cols:
        for fk_col, master_info in _FK_TO_MASTER.items():
            if fk_col not in subgroup_cols:
                continue
            master_table = master_info["master_table"]
            if master_table not in seen_masters_a:
                seen_masters_a[master_table] = (master_info["master_pk"], master_info["business_key"])

    cur = conn.cursor()

    if seen_masters_a:
        log.info(f"[LiteralResolver] Master fallback Pass A: probing {list(seen_masters_a.keys())} for '{value_upper}'")
        for master_table, (master_pk, bk_col) in seen_masters_a.items():
            try:
                cur.execute(
                    f"SELECT TOP 1 1 FROM [{master_table}] WITH (NOLOCK) "
                    f"WHERE UPPER([{bk_col}]) = ?",
                    (value_upper,)
                )
                if cur.fetchone():
                    log.info(f"[LiteralResolver] Master hit (Pass A): {master_table}.{bk_col} = '{value_upper}'")
                    return master_table, bk_col
            except Exception as e:
                log.debug(f"[LiteralResolver] Master probe failed {master_table}.{bk_col}: {e}")
                continue

    # ── Pass B: prefix-broadened fallback ────────────────────────────────────
    # Build a set of meaningful word stems from the current table names so we
    # can match against master table names that share the same domain.
    # e.g.  MFC_T_WORK_ORDER_OP  ->  stems = {'work', 'order'}
    #        MFC_M_WORK_ORDER      ->  shares 'work' and 'order'  -> candidate
    # Built dynamically at runtime from TABLE_PREFIX + TABLE_NAME_PREFIXES in .env
    # so that no module token (mfc, alm, wms, …) is ever hardcoded here.
    from core.meta_loader import get_stop_parts as _get_stop_parts
    _STOP_PARTS = _get_stop_parts()

    def _table_stems(tname: str) -> set:
        return {
            p.lower() for p in re.split(r'[_\s]+', tname)
            if len(p) >= 3 and p.lower() not in _STOP_PARTS
        }

    current_stems: set = set()
    for t in current_tables:
        current_stems.update(_table_stems(t))

    seen_masters_b: Dict[str, Tuple[str, str]] = {}
    for fk_col, master_info in _FK_TO_MASTER.items():
        master_table = master_info["master_table"]
        if master_table in seen_masters_a:          # already tried in Pass A
            continue
        if master_table in seen_masters_b:
            continue
        master_stems = _table_stems(master_table)
        if current_stems & master_stems:            # non-empty intersection
            seen_masters_b[master_table] = (master_info["master_pk"], master_info["business_key"])

    if not seen_masters_b:
        log.warning(f"[LiteralResolver] '{value_upper}' not found in any master table (Pass A + B exhausted).")
        return None

    log.info(f"[LiteralResolver] Master fallback Pass B (prefix-broadened): probing {list(seen_masters_b.keys())} for '{value_upper}'")
    for master_table, (master_pk, bk_col) in seen_masters_b.items():
        try:
            cur.execute(
                f"SELECT TOP 1 1 FROM [{master_table}] WITH (NOLOCK) "
                f"WHERE UPPER([{bk_col}]) = ?",
                (value_upper,)
            )
            if cur.fetchone():
                log.info(f"[LiteralResolver] Master hit (Pass B): {master_table}.{bk_col} = '{value_upper}'")
                return master_table, bk_col
        except Exception as e:
            log.debug(f"[LiteralResolver] Master probe failed {master_table}.{bk_col}: {e}")
            continue

    return None


# ============================================================
# FK SUBQUERY BRIDGE
# ============================================================
def build_fk_subquery_condition(
    target_table:  str,
    literal_table: str,
    literal_col:   str,
    literal_value: str,
    meta:          dict,
) -> Optional[str]:
    """
    Build WHERE subquery condition bridging literal_table to target_table
    via master PK / FK relationship. Returns condition string or None.
    """
    if target_table == literal_table:
        return None

    master_info = _MASTER_INFO.get(literal_table)
    if not master_info:
        return None

    master_pk   = master_info["pk"]
    master_bk   = master_info["bk"]
    target_cols = list(meta.get(target_table, {}).get("columns", {}).keys())

    if master_pk not in target_cols:
        return None

    condition = (
        f"{target_table}.{master_pk} = ("
        f"SELECT {master_pk} FROM {literal_table} "
        f"WHERE UPPER({master_bk}) = UPPER('{literal_value}')"
        f")"
    )
    log.info(f"[LiteralResolver] FK bridge: {target_table}.{master_pk} <- {literal_table}.{master_bk}")
    return condition


def get_fk_bridge_sql(
    target_table:  str,
    literal_table: str,
    literal_col:   str,
    literal_value: str,
    select_cols:   str,
    meta:          dict,
    intent:        str = "simple",
) -> Optional[str]:
    """Build complete SQL using FK subquery bridge. Returns None if no bridge exists."""
    condition = build_fk_subquery_condition(
        target_table, literal_table, literal_col, literal_value, meta
    )
    if not condition:
        return None

    if intent == "existence":
        sql = (
            f"SELECT CASE WHEN EXISTS("
            f"SELECT 1 FROM {target_table} WHERE {condition}"
            f") THEN 'Yes' ELSE 'No' END AS Result;"
        )
    elif intent == "aggregation":
        sql = f"SELECT COUNT(*) AS Total FROM {target_table} WHERE {condition};"
    else:
        sql = f"SELECT TOP 50 {select_cols} FROM {target_table} WHERE {condition};"

    log.info(f"[LiteralResolver] FK bridge SQL: {sql[:150]}...")
    return sql


# ============================================================
# FK ID TRANSLATION
# ============================================================
def translate_fk_ids(
    result_rows: List[dict],
    base_table:  str,
    meta:        dict,
    conn,
) -> List[dict]:
    """
    Translate FK integer ID columns in result rows to human-readable
    business key values via master table lookup. Non-fatal on errors.
    """
    if not result_rows or not _FK_TO_MASTER:
        return result_rows

    try:
        all_cols     = list(result_rows[0].keys())
        translatable = {
            col: _FK_TO_MASTER[col]
            for col in all_cols
            if col in _FK_TO_MASTER
        }

        if not translatable:
            return result_rows

        master_id_sets: Dict[str, set] = {}
        for fk_col, info in translatable.items():
            mt = info["master_table"]
            if mt not in master_id_sets:
                master_id_sets[mt] = set()
            for row in result_rows:
                val = row.get(fk_col)
                if val is not None:
                    try:
                        master_id_sets[mt].add(int(val))
                    except (ValueError, TypeError):
                        pass

        id_to_bk: Dict[str, Dict[int, str]] = {}
        cur = conn.cursor()
        for master_table, ids in master_id_sets.items():
            if not ids:
                continue
            master_info = _MASTER_INFO.get(master_table)
            if not master_info:
                continue
            pk = master_info["pk"]
            bk = master_info["bk"]
            try:
                placeholders = ",".join(str(i) for i in ids)
                cur.execute(
                    f"SELECT [{pk}], [{bk}] "
                    f"FROM [{master_table}] WITH (NOLOCK) "
                    f"WHERE [{pk}] IN ({placeholders})"
                )
                rows = cur.fetchall()
                id_to_bk[master_table] = {
                    int(r[0]): str(r[1])
                    for r in rows if r[1] is not None
                }
                log.info(f"[LiteralResolver] Fetched {len(id_to_bk[master_table])} labels from {master_table}")
            except Exception as e:
                log.warning(f"[LiteralResolver] FK translate failed for {master_table}: {e}")
                continue

        translated = []
        for row in result_rows:
            new_row = dict(row)
            for fk_col, info in translatable.items():
                mt  = info["master_table"]
                raw = new_row.get(fk_col)
                if raw is None:
                    continue
                try:
                    int_val = int(raw)
                except (ValueError, TypeError):
                    continue
                bk_val = id_to_bk.get(mt, {}).get(int_val)
                if bk_val is not None:
                    new_row[fk_col] = bk_val
            translated.append(new_row)

        return translated

    except Exception as e:
        log.warning(f"[LiteralResolver] translate_fk_ids failed, returning original: {e}", exc_info=True)
        return result_rows


# ============================================================
# UTILITY
# ============================================================
def clear_cache() -> None:
    global _VALUE_CACHE
    _VALUE_CACHE = {}
    log.info("[LiteralResolver] Cache cleared.")


def get_fk_map() -> Dict[str, Dict[str, str]]:
    """Return the FK->Master map (read-only)."""
    return dict(_FK_TO_MASTER)