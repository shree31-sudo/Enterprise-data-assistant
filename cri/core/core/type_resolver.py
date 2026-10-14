from __future__ import annotations
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("TypeResolver")

_TYPE_INDEX: Dict[str, Dict[str, Dict[str, int]]] = {}

_TYPE_CONTEXT: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}

def _load_module_prefixes() -> Tuple[str, ...]:
    """
    Build the set of module prefixes to strip from type-table names.
    Reads TABLE_NAME_PREFIXES or TABLE_PREFIX from env — same logic as
    value_searcher so both files always stay in sync.

    Used by the display-intent guard to extract a human-readable word
    from a type table name, e.g.:
        ALM_M_ALARM_TYPE  →  strip "ALM_M_"  →  "ALARM TYPE"  →  last word "type"
    """
    raw_name = os.getenv("TABLE_NAME_PREFIXES", "").strip()
    if raw_name:
        prefixes = [p.strip() for p in raw_name.split(",") if p.strip()]
        if prefixes:
            return tuple(prefixes)

    raw_code = os.getenv("TABLE_PREFIX", "").strip()
    if raw_code:
        codes = [c.strip().upper() for c in raw_code.split(",") if c.strip()]
        prefixes = (
            [f"{c}_M_" for c in codes] +
            [f"{c}_L_" for c in codes] +
            [f"{c}_T_" for c in codes]
        )
        return tuple(prefixes)

    return ("MFC_M_", "MFC_L_", "MFC_T_")

_MODULE_PREFIXES: Tuple[str, ...] = _load_module_prefixes()

def _strip_module_prefix(name: str) -> str:
    """Strip any known module prefix from a table name, replace _ with space."""
    for pfx in sorted(_MODULE_PREFIXES, key=len, reverse=True):
        if name.upper().startswith(pfx.upper()):
            return name[len(pfx):].replace("_", " ").lower()
    return name.replace("_", " ").lower()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SCHEMA DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def _discover_type_relationships(schema: dict) -> List[dict]:
    """
    Scan schema_raw.json and return a list of discovered relationships:
    [
      {
        "parent_table": "MFC_M_ITEM",
        "type_table":   "MFC_M_ITEM_TYPE",
        "fk_col":       "Item_Type_ID",      # col in parent table
        "pk_col":       "Item_Type_ID",      # PK in type table (same name)
        "name_col":     "Item_Type_Name",    # human label col in type table
      },
      ...
    ]

    Discovery rules (all based on column naming, zero hardcoding):
    1. Any table ending in _TYPE is a candidate type table.
    2. In that table, find a column ending in _Type_ID → that is the PK.
    3. The parent table = type_table with "_TYPE" stripped.
    4. The name column = first column ending in _Name or _Type_Name.
    5. Only include if the parent table actually exists in schema.
    """
    relationships = []
    known_tables  = set(schema.keys())

    for table_name, table_info in schema.items():
        if not table_name.upper().endswith("_TYPE"):
            continue

        # Get column list — handle both list-of-dicts and dict formats
        raw_cols = table_info.get("columns", [])
        if isinstance(raw_cols, dict):
            col_names = list(raw_cols.keys())
        else:
            col_names = [c["column_name"] if isinstance(c, dict) else c
                         for c in raw_cols]

        # Find the PK column (ends with _Type_ID)
        pk_col = next(
            (c for c in col_names if c.lower().endswith("_type_id")),
            None
        )
        if not pk_col:
            continue

        # Find the human label column
        name_col = next(
            (c for c in col_names
             if c.lower().endswith("_type_name") or
                (c.lower().endswith("_name") and "type" in c.lower())),
            None
        )
        # Fallback: any column ending in _Name
        if not name_col:
            name_col = next(
                (c for c in col_names if c.lower().endswith("_name")),
                None
            )
        if not name_col:
            log.debug(f"[TypeResolver] {table_name}: no _Name column found, skipping.")
            continue

        # Derive parent table — strip _TYPE suffix
        parent_table = table_name[:-5]  # "MFC_M_ITEM_TYPE" → "MFC_M_ITEM"

        # Only proceed if the parent table actually exists
        if parent_table not in known_tables:
            # Try case-insensitive match
            parent_table = next(
                (t for t in known_tables if t.upper() == parent_table.upper()),
                None
            )
            if not parent_table:
                log.debug(
                    f"[TypeResolver] {table_name}: parent '{table_name[:-5]}' "
                    f"not in schema, skipping."
                )
                continue

        relationships.append({
            "parent_table": parent_table,
            "type_table":   table_name,
            "fk_col":       pk_col,    
            "pk_col":       pk_col,
            "name_col":     name_col,
        })
        log.info(
            f"[TypeResolver] Discovered: {parent_table}.{pk_col} "
            f"→ {table_name}.{name_col}"
        )

    return relationships


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LABEL NORMALISATION
# Generates spelling variants so fuzzy matching catches plurals,
# abbreviations, and hyphen/space differences.
# ═════════════════════════════════════════════════════════════════════════════

def _normalise_label(raw: str) -> str:
    """Lowercase + strip extra whitespace."""
    return raw.lower().strip()


def _generate_variants(label: str) -> List[str]:
    """
    Given "Raw Material" produce safe variants for matching.

    RULE: Single-word variants are EXCLUDED unless the label itself is
    a single word. Words like "order", "work", "material" appearing alone
    in a question score 1.0 and cause massive false-positive type filters
    (e.g. "show order wocnc0101" would match "MFG Work Order" via "order").

    Safe variants kept:
      - The full label as-is            "raw material"
      - Plural/singular of full label   "raw materials"
      - Hyphen/space/concat forms       "raw-material", "rawmaterial"
      - Multi-word subphrases only (2+ words) from compound labels
    """
    base = _normalise_label(label)
    variants = {base}

    # Plural / singular of the full label
    if base.endswith("s"):
        variants.add(base[:-1])
    else:
        variants.add(base + "s")

    # Hyphen ↔ space ↔ concat (safe — still discriminative as full phrase)
    variants.add(base.replace(" ", "-"))
    variants.add(base.replace("-", " "))
    variants.add(base.replace(" ", ""))
    variants.add(base.replace("-", ""))

    # Multi-word subphrases ONLY — never single isolated words.
    # A single word like "order" or "work" is too common to be discriminative.
    words = re.split(r"[\s\-_/]+", base)
    if len(words) >= 3:
        # For 3+ word labels, also allow 2-word subphrases (adjacent pairs)
        for i in range(len(words) - 1):
            pair = words[i] + " " + words[i + 1]
            if all(len(w) > 2 for w in [words[i], words[i+1]]):
                variants.add(pair)
                variants.add(pair + "s")

    return list(variants)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STARTUP LOADER
# Call this ONCE at application startup with an open DB connection.
# ═════════════════════════════════════════════════════════════════════════════

def load_type_mappings(conn, schema: dict, db_schema: str = "dbo") -> None:
    """
    Read every discovered *_TYPE table from the DB and populate
    _TYPE_INDEX and _TYPE_CONTEXT.

    Parameters
    ----------
    conn      : open DB connection
    schema    : the parsed schema_raw.json dict
    db_schema : DB schema prefix (default "dbo" for SQL Server).
                Use "public" for PostgreSQL, "" for MySQL/SQLite.

    This function is idempotent — safe to call multiple times (e.g. on refresh).
    On any DB error for a single table it logs a warning and continues —
    a partial index is better than no index.
    """
    global _TYPE_INDEX, _TYPE_CONTEXT
    _TYPE_INDEX   = {}
    _TYPE_CONTEXT = {}

    relationships = _discover_type_relationships(schema)
    if not relationships:
        log.warning("[TypeResolver] No _TYPE relationships discovered — check schema_raw.json.")
        return

    cur = conn.cursor()
    loaded, skipped = 0, 0

    for rel in relationships:
        parent = rel["parent_table"]
        fk_col = rel["fk_col"]
        t_tbl  = rel["type_table"]
        pk_col = rel["pk_col"]
        nm_col = rel["name_col"]

        try:
            # Schema prefix is configurable — not hardcoded to dbo
            tbl_ref = f"[{db_schema}].[{t_tbl}]" if db_schema else f"[{t_tbl}]"
            cur.execute(
                f"SELECT [{pk_col}], [{nm_col}] "
                f"FROM {tbl_ref} "
                f"WHERE [{nm_col}] IS NOT NULL"
            )
            rows = cur.fetchall()
        except Exception as e:
            log.warning(f"[TypeResolver] Could not read {t_tbl}: {e}")
            skipped += 1
            continue

        if not rows:
            log.debug(f"[TypeResolver] {t_tbl} returned 0 rows, skipping.")
            continue

        # Build label → id map with all variants
        label_to_id: Dict[str, int] = {}
        raw_rows: List[Tuple[int, str]] = []

        for type_id, type_name in rows:
            if type_id is None or type_name is None:
                continue
            try:
                tid = int(type_id)
            except (ValueError, TypeError):
                continue

            raw_rows.append((tid, str(type_name).strip()))

            for variant in _generate_variants(str(type_name)):
                # First-write wins — more specific names take priority
                if variant not in label_to_id:
                    label_to_id[variant] = tid

        # Store in module cache
        _TYPE_INDEX.setdefault(parent, {})[fk_col]   = label_to_id
        _TYPE_CONTEXT.setdefault(parent, {})[fk_col] = raw_rows
        loaded += 1

        log.info(
            f"[TypeResolver] Loaded {t_tbl}: "
            f"{len(raw_rows)} types, {len(label_to_id)} label variants "
            f"→ {parent}.{fk_col}"
        )

    log.info(
        f"[TypeResolver] Startup complete: "
        f"{loaded} _TYPE tables loaded, {skipped} skipped."
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUERY-TIME RESOLVER
# Called during handle_query to find a type filter from the question.
# ═════════════════════════════════════════════════════════════════════════════

def _token_overlap_score(phrase: str, question_lower: str) -> float:
    """
    Score how well a phrase matches the question.
    Returns 0.0–1.0.

    Scoring:
    1. Exact substring match (whole phrase)          → 1.0
    2. All words of phrase present as whole words    → 0.85
    3. Most words found as whole words               → proportional * 0.6

    Single-word phrases are penalised — they match too broadly.
    A 1-word variant scoring 1.0 would fire on ANY question containing
    that word, regardless of intent. We cap single-word scores at 0.50
    so they never reach min_score=0.6 (the call site in resolve_lov_matches).

    This specifically prevents "types" / "type" in "show items with their types"
    from scoring high enough to inject a spurious Item_Type_ID filter when the
    user is asking to display a joined _TYPE table, not filter by a type value.
    """
    phrase_words = re.split(r"[\s\-_]+", phrase.strip())
    phrase_words = [w for w in phrase_words if w]

    # Single-word phrase — use whole-word boundary check, cap at 0.50
    # 0.50 < min_score=0.6, so single-word matches are always suppressed.
    if len(phrase_words) == 1:
        w = phrase_words[0]
        if re.search(rf"\b{re.escape(w)}\b", question_lower):
            return 0.50   # present but capped — won't reach min_score=0.6
        return 0.0

    # Multi-word phrase — exact substring is strongest signal
    if phrase in question_lower:
        return 1.0

    # All words present as whole words
    found = sum(
        1 for w in phrase_words
        if re.search(rf"\b{re.escape(w)}\b", question_lower)
    )
    ratio = found / len(phrase_words)
    if ratio == 1.0:
        return 0.85
    return ratio * 0.6


def get_type_filter(
    base_table: str,
    question:   str,
    min_score:  float = 0.75,
) -> Optional[Dict]:
    """
    Given a base table and a user question, find if the question mentions
    a type category that maps to a FK column on that table.

    Returns
    -------
    {
        "col":        "Item_Type_ID",
        "value":      6,
        "label":      "Raw Material",
        "type_table": "MFC_M_ITEM_TYPE",
    }
    or None if no match found.

    Matching is fuzzy — "raw materials", "RM items", "raw" etc.
    all match "Raw Material" → 6.

    NOTE: Returns None immediately when the question is a join-display intent
    (e.g. "show items with their types") — the word "types" refers to the
    joined _TYPE table, not a filter value.
    """
    # ── Display-intent guard ─────────────────────────────────────────────────
    # If the question contains a display signal AND the _TYPE table name for
    # this base_table is referenced, this is a join/display question —
    # do not inject a type filter.
    _DISPLAY_RE = re.compile(
        r'\b(with\s+their|along\s+with|and\s+their|including|show.*and|list.*with)\b',
        re.IGNORECASE,
    )
    _type_table_word = _strip_module_prefix(base_table + "_TYPE")
    _type_word       = _type_table_word.split()[-1] if _type_table_word else ""
    _q_lower_guard   = question.lower()
    if (
        _DISPLAY_RE.search(question)
        and _type_word
        and re.search(rf"\b{re.escape(_type_word)}s?\b", _q_lower_guard)
    ):
        log.info(
            f"[TypeResolver] Display-intent guard fired for '{question[:60]}' "
            f"on {base_table} — skipping type filter."
        )
        return None
    # ─────────────────────────────────────────────────────────────────────────
    table_map = _TYPE_INDEX.get(base_table, {})
    if not table_map:
        # Try uppercase key
        table_map = _TYPE_INDEX.get(base_table.upper(), {})
    if not table_map:
        return None

    q_lower = question.lower()
    best_score  = 0.0
    best_match  = None

    for fk_col, label_map in table_map.items():
        for variant, type_id in label_map.items():
            score = _token_overlap_score(variant, q_lower)
            if score > best_score:
                best_score = score
                # Recover the original raw name for logging
                context_rows = _TYPE_CONTEXT.get(base_table, {}).get(fk_col, [])
                raw_name = next(
                    (name for tid, name in context_rows if tid == type_id),
                    variant
                )
                # Derive type_table name from the relationship cache
                type_table = base_table + "_TYPE"
                best_match = {
                    "col":        fk_col,
                    "value":      type_id,
                    "label":      raw_name,
                    "type_table": type_table,
                    "score":      score,
                }

    if best_match and best_score >= min_score:
        log.info(
            f"[TypeResolver] Match for '{question[:60]}': "
            f"table={base_table} col={best_match['col']} "
            f"label='{best_match['label']}' id={best_match['value']} "
            f"score={best_score:.2f}"
        )
        return best_match

    return None


def get_type_filters_all(
    base_table: str,
    question:   str,
    min_score:  float = 0.75,
) -> List[Dict]:
    """
    Like get_type_filter but returns ALL matches above min_score,
    not just the best one. Useful when a question has multiple type filters.

    E.g. "show raw material items with tooling BOM"
    might match both Item_Type_ID and BOM_Type_ID on different tables.

    NOTE: Returns [] immediately when the question is a join-display intent —
    same guard as get_type_filter.
    """
    # ── Display-intent guard (same logic as get_type_filter) ─────────────────
    _DISPLAY_RE = re.compile(
        r'\b(with\s+their|along\s+with|and\s+their|including|show.*and|list.*with)\b',
        re.IGNORECASE,
    )
    _type_table_word = _strip_module_prefix(base_table + "_TYPE")
    _type_word       = _type_table_word.split()[-1] if _type_table_word else ""
    _q_lower_guard   = question.lower()
    if (
        _DISPLAY_RE.search(question)
        and _type_word
        and re.search(rf"\b{re.escape(_type_word)}s?\b", _q_lower_guard)
    ):
        log.info(
            f"[TypeResolver] Display-intent guard fired (all) for '{question[:60]}' "
            f"on {base_table} — returning no type filters."
        )
        return []
    # ─────────────────────────────────────────────────────────────────────────
    table_map = _TYPE_INDEX.get(base_table, {}) or _TYPE_INDEX.get(base_table.upper(), {})
    if not table_map:
        return []

    q_lower    = question.lower()
    col_best:  Dict[str, Tuple[float, Dict]] = {}

    for fk_col, label_map in table_map.items():
        for variant, type_id in label_map.items():
            score = _token_overlap_score(variant, q_lower)
            if score >= min_score:
                prev_score = col_best.get(fk_col, (0.0, None))[0]
                if score > prev_score:
                    context_rows = _TYPE_CONTEXT.get(base_table, {}).get(fk_col, [])
                    raw_name = next(
                        (name for tid, name in context_rows if tid == type_id),
                        variant
                    )
                    type_table = base_table + "_TYPE"
                    col_best[fk_col] = (score, {
                        "col":        fk_col,
                        "value":      type_id,
                        "label":      raw_name,
                        "type_table": type_table,
                        "score":      score,
                    })

    results = [v for _, v in col_best.values()]
    if results:
        log.info(
            f"[TypeResolver] {len(results)} type filter(s) for "
            f"'{question[:60]}' on {base_table}: "
            + ", ".join(f"{r['col']}={r['value']}({r['label']})" for r in results)
        )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLM PROMPT CONTEXT
# Returns a formatted text block that gets injected into the LLM prompt
# so the model knows what type IDs to use.
# ═════════════════════════════════════════════════════════════════════════════

def get_type_lov_context(base_table: str) -> str:
    """
    Returns a formatted string for injection into build_lov_context_block().

    Example output:
    TYPE VALUES for MFC_M_ITEM — use numeric IDs ONLY:
      Item_Type_ID (from MFC_M_ITEM_TYPE):
        Raw Material → 6  |  Finished Goods → 1  |  Machine → 3  |  Tool → 4

    Returns "" if no type data available for this table.
    """
    context_map = _TYPE_CONTEXT.get(base_table, {}) or _TYPE_CONTEXT.get(base_table.upper(), {})
    if not context_map:
        return ""

    lines = [f"TYPE VALUES for {base_table} — use numeric IDs ONLY:"]
    for fk_col, rows in context_map.items():
        if not rows:
            continue
        type_table = base_table + "_TYPE"
        entries    = "  |  ".join(f"{name} → {tid}" for tid, name in rows)
        lines.append(f"  {fk_col} (from {type_table}): {entries}")

    return "\n".join(lines) if len(lines) > 1 else ""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def is_loaded() -> bool:
    """Return True if load_type_mappings() has been called and produced data."""
    return bool(_TYPE_INDEX)


def get_loaded_tables() -> List[str]:
    """Return list of parent tables that have type mappings loaded."""
    return list(_TYPE_INDEX.keys())