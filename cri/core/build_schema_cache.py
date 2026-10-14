from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

log = logging.getLogger("build_schema_cache")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)


def _active_prefixes() -> Tuple[str, ...]:
    raw = os.getenv("TABLE_PREFIX", "").strip()
    prefixes = tuple(p.strip() for p in raw.split(",") if p.strip())
    if not prefixes:
        log.error(
            "TABLE_PREFIX not set in environment. "
            "Set it as a comma-separated list, e.g. TABLE_PREFIX=MFC,PRD,ACM,..."
        )
        sys.exit(1)
    return prefixes

_PREFIX_STOPWORDS: Set[str] = {p.lower() for p in _active_prefixes()}

# ── Constants matching chatbot_backend.py ─────────────────────────────────────
AUDIT_COL_SUFFIXES: Tuple[str, ...] = (
    "cdate", "cuser", "udate", "uuser",
    "tenant_id", "notes", "remarks", "description",
    "lami_cdate", "lami_cuser", "lami_udate", "lami_uuser",
    "lami_status", "lami_inactive_reason",
)
DATE_SUFFIXES: Tuple[str, ...] = (
    "date", "datetime", "date_time", "dt",
    "created_date", "cdate", "start_date", "end_date", "modified_date",
    "lami_cdate", "lami_udate",
)
STATUS_KEYWORDS: Tuple[str, ...] = (
    "status", "state", "flag", "type", "mode",
)
BUSINESS_KEYWORDS: Tuple[str, ...] = (
    "name", "number", "code", "description", "title",
    "prefix", "no", "ref", "label",
)


# ═════════════════════════════════════════════════════════════════════════════
# DB helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_dsn() -> str:
    if os.getenv("DB_DSN"):
        return os.environ["DB_DSN"]
    driver   = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    server   = os.getenv("DB_SERVER", "10.225.30.26")
    database = os.getenv("DB_DATABASE", "LAMI_DB")
    uid      = os.getenv("DB_UID", "sa")
    pwd      = os.getenv("DB_PWD", "sql@123")
    if not server or not database:
        log.error(
            "DB connection not configured. "
            "Set DB_DSN or DB_SERVER + DB_DATABASE + DB_UID + DB_PWD."
        )
        sys.exit(1)
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        "TrustServerCertificate=yes;"
    )


def _connect(dsn: str):
    try:
        import pyodbc
        conn = pyodbc.connect(dsn, timeout=30)
        conn.autocommit = True
        return conn
    except ImportError:
        log.error("pyodbc not installed. Run: pip install pyodbc")
        sys.exit(1)
    except Exception as exc:
        log.error("DB connection failed: %s", exc)
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Column classification helpers
# ═════════════════════════════════════════════════════════════════════════════

def _is_audit(col: str) -> bool:
    cl = col.lower()
    return any(cl.endswith(s) for s in AUDIT_COL_SUFFIXES)


def _is_date_col(col: str, sql_type: str) -> bool:
    cl = col.lower()
    type_lower = sql_type.lower()
    if any(t in type_lower for t in ("date", "time")):
        return True
    return any(cl.endswith(s) for s in DATE_SUFFIXES)


def _is_status_col(col: str) -> bool:
    cl = col.lower()
    return any(kw in cl for kw in STATUS_KEYWORDS)


def _is_business_col(col: str) -> bool:
    cl = col.lower()
    return any(cl.endswith(kw) or cl.startswith(kw) for kw in BUSINESS_KEYWORDS)


def _tokenize(text: str) -> List[str]:
    """
    Split snake_case identifier into meaningful tokens (≥ 3 chars).
    Filters out common stopwords AND all module prefix names so they
    don't pollute search_tokens.
    """
    _GENERIC_STOPWORDS = {"the", "and", "for"}
    parts = re.split(r"[_\s]+", text.lower())
    return [
        p for p in parts
        if len(p) >= 3
        and p not in _GENERIC_STOPWORDS
        and p not in _PREFIX_STOPWORDS
    ]


# ═════════════════════════════════════════════════════════════════════════════
# SQL Server introspection queries
# ═════════════════════════════════════════════════════════════════════════════

_SQL_ALL_TABLES = """
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
  AND TABLE_NAME LIKE ?
ORDER BY TABLE_NAME
"""

_SQL_COLUMNS = """
SELECT
    c.COLUMN_NAME,
    c.DATA_TYPE,
    c.IS_NULLABLE,
    c.ORDINAL_POSITION,
    CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS IS_PK
FROM INFORMATION_SCHEMA.COLUMNS c
LEFT JOIN (
    SELECT ku.COLUMN_NAME
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
      ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
     AND tc.TABLE_NAME      = ku.TABLE_NAME
    WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
      AND tc.TABLE_NAME = ?
) pk ON pk.COLUMN_NAME = c.COLUMN_NAME
WHERE c.TABLE_NAME = ?
ORDER BY c.ORDINAL_POSITION
"""

_SQL_FK = """
SELECT
    fk.name                           AS fk_name,
    tp.name                           AS parent_table,
    cp.name                           AS parent_col,
    tr.name                           AS ref_table,
    cr.name                           AS ref_col
FROM sys.foreign_keys fk
JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
JOIN sys.tables tp  ON tp.object_id  = fkc.parent_object_id
JOIN sys.columns cp ON cp.object_id  = fkc.parent_object_id
                   AND cp.column_id  = fkc.parent_column_id
JOIN sys.tables tr  ON tr.object_id  = fkc.referenced_object_id
JOIN sys.columns cr ON cr.object_id  = fkc.referenced_object_id
                   AND cr.column_id  = fkc.referenced_column_id
WHERE tp.name LIKE ?
ORDER BY tp.name, fk.name
"""


# ═════════════════════════════════════════════════════════════════════════════
# Core extraction
# ═════════════════════════════════════════════════════════════════════════════

def fetch_tables_for_prefix(conn, prefix: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(_SQL_ALL_TABLES, (f"{prefix}%",))
    return [row[0] for row in cur.fetchall()]


def fetch_all_tables(conn, prefixes: Tuple[str, ...]) -> List[str]:
    """
    Discover all tables across every prefix. Deduplicates in case of overlap.
    """
    seen:   Set[str]  = set()
    result: List[str] = []
    for prefix in prefixes:
        tables = fetch_tables_for_prefix(conn, prefix)
        for t in tables:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def fetch_columns(conn, table: str) -> List[Dict]:
    """Return list of {name, data_type, nullable, is_pk}."""
    cur = conn.cursor()
    cur.execute(_SQL_COLUMNS, (table, table))
    rows = cur.fetchall()
    return [
        {
            "name":      row[0],
            "data_type": row[1],
            "nullable":  row[2] == "YES",
            "is_pk":     bool(row[4]),
        }
        for row in rows
    ]


def fetch_all_fk(conn, prefixes: Tuple[str, ...]) -> Dict[str, List[Dict]]:
    """
    Fetch FK relationships for ALL prefixes in one pass.
    Returns {parent_table: [{column, ref_table, ref_col}, ...]}
    """
    result: Dict[str, List[Dict]] = {}
    cur = conn.cursor()
    for prefix in prefixes:
        cur.execute(_SQL_FK, (f"{prefix}%",))
        for _, parent_tbl, parent_col, ref_tbl, ref_col in cur.fetchall():
            result.setdefault(parent_tbl, []).append({
                "column":    parent_col,
                "ref_table": ref_tbl,
                "ref_col":   ref_col,
            })
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Join path precomputation  (BFS over FK graph, depth ≤ 3)
# ═════════════════════════════════════════════════════════════════════════════

def _build_fk_graph(
    all_fk: Dict[str, List[Dict]],
    all_tables: Set[str],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Bidirectional FK adjacency list.
    {table: [(neighbour_table, join_col_on_table, join_col_on_neighbour), ...]}
    """
    graph: Dict[str, List[Tuple[str, str, str]]] = {t: [] for t in all_tables}
    for parent, fk_list in all_fk.items():
        if parent not in all_tables:
            continue
        for fk in fk_list:
            ref = fk["ref_table"]
            if ref not in all_tables:
                continue
            graph[parent].append((ref, fk["column"], fk["ref_col"]))
            graph[ref].append((parent, fk["ref_col"], fk["column"]))
    return graph


def _bfs_join_paths(
    graph: Dict[str, List[Tuple[str, str, str]]],
    all_tables: Set[str],
    max_depth: int = 3,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    For every table pair (A, B) reachable within max_depth hops,
    store the shortest join path as a list of steps:
        [{from_table, from_col, to_table, to_col}, ...]

    Result: {base_table: {target_table: [steps]}}
    """
    paths: Dict[str, Dict[str, List[Dict]]] = {}

    for start in all_tables:
        paths[start] = {}
        queue = [(start, [])]
        visited: Set[str] = {start}

        while queue:
            cur_tbl, cur_path = queue.pop(0)
            if len(cur_path) >= max_depth:
                continue
            for (nbr, cur_col, nbr_col) in graph.get(cur_tbl, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                step = {
                    "from_table": cur_tbl,
                    "from_col":   cur_col,
                    "to_table":   nbr,
                    "to_col":     nbr_col,
                }
                new_path = cur_path + [step]
                paths[start][nbr] = new_path
                queue.append((nbr, new_path))

    return paths


# ═════════════════════════════════════════════════════════════════════════════
# Schema cache builder
# ═════════════════════════════════════════════════════════════════════════════

def build_full_schema(
    conn,
    tables: List[str],
    prefixes: Tuple[str, ...],
) -> Dict[str, Dict]:
    log.info("Fetching FK relationships for %d prefix(es)...", len(prefixes))
    all_fk = fetch_all_fk(conn, prefixes)
    log.info("  → %d FK relationships found across all modules", sum(len(v) for v in all_fk.values()))

    log.info("Extracting column metadata for %d tables...", len(tables))
    table_set = set(tables)
    raw: Dict[str, Dict] = {}

    for table in tables:
        cols_info = fetch_columns(conn, table)
        if not cols_info:
            log.warning("  ⚠  %s — no columns returned, skipping.", table)
            continue

        pk_cols:       List[str] = []
        date_cols:     List[str] = []
        status_cols:   List[str] = []
        business_cols: List[str] = []
        audit_cols:    List[str] = []
        all_cols:      List[str] = []
        col_meta:      Dict[str, Dict] = {}

        for ci in cols_info:
            name = ci["name"]
            all_cols.append(name)
            col_meta[name] = {
                "data_type": ci["data_type"],
                "nullable":  ci["nullable"],
                "is_pk":     ci["is_pk"],
            }
            if ci["is_pk"]:
                pk_cols.append(name)
            if _is_audit(name):
                audit_cols.append(name)
                continue
            if _is_date_col(name, ci["data_type"]):
                date_cols.append(name)
            if _is_status_col(name):
                status_cols.append(name)
            if _is_business_col(name):
                business_cols.append(name)

        fk_list = all_fk.get(table, [])

        token_set: Set[str] = set(_tokenize(table))
        for col in all_cols:
            if not _is_audit(col):
                token_set.update(_tokenize(col))
        search_tokens = sorted(token_set)

        raw[table] = {
            "columns":          all_cols,
            "pk":               pk_cols,
            "fk":               fk_list,
            "date_columns":     date_cols,
            "status_columns":   status_cols,
            "business_columns": business_cols,
            "audit_columns":    audit_cols,
            "col_meta":         col_meta,
            "search_tokens":    search_tokens,
            "join_paths":       {},
        }
        log.info(
            "  ✓  %-50s  %d cols  pk=%s  fk=%d",
            table, len(all_cols), pk_cols, len(fk_list),
        )

    # ── Precompute join paths via BFS ──────────────────────────────────────
    log.info("Precomputing join paths (BFS depth ≤ 3) across %d tables...", len(raw))
    fk_graph   = _build_fk_graph(all_fk, table_set)
    join_paths = _bfs_join_paths(fk_graph, table_set, max_depth=3)

    for table, paths in join_paths.items():
        if table in raw:
            raw[table]["join_paths"] = paths

    log.info("Join path computation complete.")
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# Atomic JSON write
# ═════════════════════════════════════════════════════════════════════════════

def _write_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    log.info("✅  Written → %s  (%d tables, %.1f KB)", path, len(data), path.stat().st_size / 1024)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full schema cache JSON for ALL modules from SQL Server."
    )
    parser.add_argument("--dsn",        help="Full ODBC connection string (overrides env vars)")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("SCHEMA_OUTPUT_DIR", "data"),
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        help="Explicit table list (overrides TABLE_PREFIX discovery — for testing only)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir or "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "full_schema_cache.json"

    dsn = args.dsn or _build_dsn()
    log.info("Connecting to SQL Server...")
    conn = _connect(dsn)
    log.info("Connected ✓")

    t0 = time.perf_counter()
    active_prefixes = _active_prefixes()

    if args.tables:
        tables = args.tables
        log.info("Using explicit table list: %d tables", len(tables))
        prefixes_used = active_prefixes
    else:
        log.info(
            "Discovering tables across %d prefix(es): %s ...",
            len(active_prefixes),
            ", ".join(active_prefixes),
        )
        tables = fetch_all_tables(conn, active_prefixes)
        log.info("Discovered %d tables total across all modules", len(tables))
        prefixes_used = active_prefixes

    if not tables:
        log.error("No tables found for any prefix. Check DB connection or TABLE_PREFIX.")
        sys.exit(1)

    full_schema = build_full_schema(conn, tables, prefixes_used)
    conn.close()

    _write_json(full_schema, full_path)

    elapsed = time.perf_counter() - t0
    log.info(
        "Done in %.1fs — %d tables extracted → %s",
        elapsed, len(full_schema), full_path.name,
    )

    # ── Summary by module ─────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Tables extracted per module:")
    from collections import Counter
    module_counts: Counter = Counter()
    for tname in full_schema:
        matched = next(
            (p for p in sorted(active_prefixes, key=len, reverse=True) if tname.upper().startswith(p.upper())),
            "OTHER",
        )
        module_counts[matched] += 1
    for mod, count in sorted(module_counts.items()):
        log.info("  %-25s  %d tables", mod, count)
    log.info("─" * 60)


if __name__ == "__main__":
    main()