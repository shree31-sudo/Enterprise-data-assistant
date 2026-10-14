import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── Allow running from any working directory ─────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_master_cache")

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA        = _PROJECT_ROOT / "data"
_GRAPHS_DIR  = _DATA / "join_graphs"          # *_join_graph.json files live here
_CACHE_PATH  = _DATA / "master_translation_cache.json"
_CACHE_TTL_DAYS = int(os.getenv("MASTER_CACHE_TTL_DAYS", "15"))

# ── Table classification ───────────────────────────────────────────────────────
# Tables with these suffixes are operational/log tables, not master/lookup tables.
# Applied to ALL prefixes, not just MFC.
_SKIP_SUFFIXES: tuple = (
    "_LOG", "_HISTORY", "_AUDIT", "_ARCHIVE",
    "_HIST", "_TRACK", "_TRACKING", "_TEMP", "_TMP",
)

def _is_master_table(table_name: str) -> bool:
    """
    A table is a candidate master/lookup table if it does NOT end with a
    known operational suffix.  We no longer restrict to MFC_L_/MFC_M_ —
    any prefix's lookup table can serve as a translation source.
    """
    up = table_name.upper()
    return not any(up.endswith(sfx) for sfx in _SKIP_SUFFIXES)


# ── DB connection ─────────────────────────────────────────────────────────────
def _get_conn():
    from config.db import get_db_connection
    return get_db_connection()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pick_name_column(table_name: str, all_cols: list) -> Optional[str]:
    """
    Heuristic: prefer columns whose name contains 'name', 'desc', 'code',
    'title', 'label' (in that order).  Falls back to None if nothing fits.
    """
    priority_keywords = ["name", "desc", "code", "title", "label"]
    cols_lower = {c.lower(): c for c in all_cols}

    for kw in priority_keywords:
        for cl, orig in cols_lower.items():
            if cl.endswith("_" + kw) or cl == kw:
                return orig
        for cl, orig in cols_lower.items():
            if kw in cl:
                return orig

    return None


def _fetch_id_name_map(conn, table: str, pk_col: str, name_col: str) -> dict:
    """
    Fetch {id_value → display_name} from a single master table.
    Returns {} on any error so one bad table never breaks the whole run.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT [{pk_col}], [{name_col}] FROM [{table}] WITH (NOLOCK)"
        )
        rows = cur.fetchall()
        return {str(row[0]).strip(): str(row[1]).strip() for row in rows if row[0] is not None}
    except Exception as exc:
        log.warning("[fetch] %s — skipping: %s", table, exc)
        return {}


# ── FK resolution from join_graphs ───────────────────────────────────────────

def _load_fk_col_to_master_from_graphs() -> Dict[str, str]:
    """
    Walk every *_join_graph.json in data/join_graphs/ and extract
    from_column → to_table mappings from all 1-hop join_paths steps.

    This replaces the old mfc_narrow.json + mfc.json approach and covers
    all 29 prefixes automatically.

    Logic:
      Each step in join_paths is:
        { from_table, from_column, to_table, to_column }
      from_column is the FK side; to_table is the master being referenced.
      We register: FK_COLUMN_UPPER → to_table  (first seen wins, same as before).
    """
    graph_files = sorted(_GRAPHS_DIR.glob("*_join_graph.json"))
    if not graph_files:
        log.error("No *_join_graph.json files found in %s — run build_join_graph.py first.", _GRAPHS_DIR)
        sys.exit(1)

    log.info("Resolving FK mappings from %d join graph file(s) …", len(graph_files))

    fk_col_to_master: Dict[str, str] = {}

    for gf in graph_files:
        data       = json.loads(gf.read_text(encoding="utf-8"))
        prefix     = data.get("_meta", {}).get("prefix", gf.stem)
        join_paths = data.get("join_paths", {})

        for source, targets in join_paths.items():
            for target, steps in targets.items():
                # Only look at 1-hop steps (direct FK relationships)
                if len(steps) != 1:
                    continue
                step     = steps[0]
                from_col = step.get("from_column", "").upper()
                to_table = step.get("to_table",    "")

                if not from_col or not to_table:
                    continue
                if not _is_master_table(to_table):
                    continue
                # First seen wins — enforced_fk edges appear first in the graph
                # because build_join_graph.py merges in enforced → implicit_pk → implicit_ui order
                if from_col not in fk_col_to_master:
                    fk_col_to_master[from_col] = to_table

        log.info("  [%s] running total FK mappings: %d", prefix, len(fk_col_to_master))

    log.info("FK column→master table mappings resolved: %d", len(fk_col_to_master))
    return fk_col_to_master


# ══════════════════════════════════════════════════════════════════════════════
def build_cache(dry_run: bool = False) -> dict:

    # ── Load full_schema_cache (still needed for PK/column info) ──────────
    log.info("Loading full_schema_cache.json …")
    full_schema: dict = json.loads((_DATA / "full_schema_cache.json").read_text(encoding="utf-8"))

    # ── Load LOV data (still needed, no change) ────────────────────────────
    log.info("Loading lov_mapping_dynamic.json …")
    lov_data: dict = json.loads((_DATA / "lov_mapping_dynamic.json").read_text(encoding="utf-8"))

    # ── Build FK column → master table mapping from join graphs ───────────
    # Replaces the old mfc_narrow.json + mfc.json walk entirely.
    fk_col_to_master = _load_fk_col_to_master_from_graphs()

    # ── Supplement: any master table PK not yet covered ───────────────────
    # Walk full_schema_cache for tables that pass _is_master_table() and
    # register their PK column if not already mapped.  Same logic as before,
    # now prefix-agnostic.
    for tname, tinfo in full_schema.items():
        if not _is_master_table(tname):
            continue
        for pk_col in tinfo.get("pk", []):
            pk_upper = pk_col.upper()
            if pk_upper not in fk_col_to_master:
                fk_col_to_master[pk_upper] = tname

    log.info("FK mappings after full_schema supplement: %d", len(fk_col_to_master))

    # ── Determine which master tables actually need to be queried ──────────
    needed_tables: Set[str] = set(fk_col_to_master.values())
    log.info("Master tables to query: %d", len(needed_tables))

    # ── Build LOV cache from lov_mapping_dynamic.json (no DB needed) ──────
    lov_cache: Dict[str, dict] = {}
    for _tbl, col_map in lov_data.items():
        for col, mapping in col_map.items():
            col_upper = col.upper()
            if col_upper not in lov_cache:
                lov_cache[col_upper] = {}
            for label, db_val in mapping.items():
                lov_cache[col_upper][str(db_val)] = label

    log.info("LOV cache built: %d column(s)", len(lov_cache))

    # ── Query DB for FK and TYPE translations ─────────────────────────────
    fk_cache:   Dict[str, dict] = {}
    type_cache: Dict[str, dict] = {}

    if dry_run:
        log.info("[DRY RUN] Skipping DB queries. Would query %d table(s).", len(needed_tables))
        for tname in sorted(needed_tables):
            log.info("  → %s", tname)
    else:
        try:
            conn = _get_conn().__enter__()
        except Exception as exc:
            log.error("Cannot connect to DB: %s", exc)
            sys.exit(1)

        try:
            for tname in sorted(needed_tables):
                tinfo    = full_schema.get(tname, {})
                all_cols = tinfo.get("columns", [])
                pk_list  = tinfo.get("pk", [])

                if not pk_list:
                    log.warning("[skip] %s — no PK found in full_schema_cache", tname)
                    continue

                pk_col   = pk_list[0]
                name_col = _pick_name_column(tname, all_cols)

                if not name_col:
                    log.warning("[skip] %s — no suitable name column found", tname)
                    continue

                id_map = _fetch_id_name_map(conn, tname, pk_col, name_col)
                if not id_map:
                    log.warning("[empty] %s — no rows returned", tname)
                    continue

                log.info("[ok] %s → pk=%s  name=%s  rows=%d", tname, pk_col, name_col, len(id_map))

                # Register under every FK col that points to this table
                for fk_col, master in fk_col_to_master.items():
                    if master == tname:
                        fk_cache[fk_col] = id_map

                # Also register as TYPE cache if table is a type/category table
                pk_upper = pk_col.upper()
                if any(kw in tname.upper() for kw in ["_TYPE", "_CATEGORY", "_KIND"]):
                    type_cache[pk_upper] = id_map

        finally:
            try:
                conn.__exit__(None, None, None)
            except Exception:
                pass

    # ── Assemble final payload ─────────────────────────────────────────────
    now     = datetime.utcnow()
    expires = now + timedelta(days=_CACHE_TTL_DAYS)

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "expires_at":   expires.isoformat(timespec="seconds"),
        "ttl_days":     _CACHE_TTL_DAYS,
        "fk":           fk_cache,
        "lov":          lov_cache,
        "type":         type_cache,
        "stats": {
            "fk_columns":            len(fk_cache),
            "lov_columns":           len(lov_cache),
            "type_columns":          len(type_cache),
            "master_tables_queried": len(needed_tables),
        },
    }

    if dry_run:
        log.info("[DRY RUN] Cache NOT written. Stats: %s", payload["stats"])
        return payload

    _DATA.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Cache written → %s  |  stats: %s", _CACHE_PATH, payload["stats"])
    return payload


# ── CLI ───────────────────────────────────────────────────────────────────────
def _is_cache_stale() -> bool:
    if not _CACHE_PATH.exists():
        return True
    try:
        data    = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(data.get("expires_at", "2000-01-01"))
        return datetime.utcnow() > expires
    except Exception:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master translation cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be cached; no DB queries, no write.")
    parser.add_argument("--force",   action="store_true", help="Rebuild even if cache is still fresh.")
    args = parser.parse_args()

    if not args.force and not args.dry_run and not _is_cache_stale():
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        log.info(
            "Cache is still fresh (expires %s). Use --force to rebuild.",
            data.get("expires_at"),
        )
        sys.exit(0)

    build_cache(dry_run=args.dry_run)