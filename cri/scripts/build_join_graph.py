from __future__ import annotations

import json
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Allow running from project root without installing the package ────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_join_graph")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "join_graphs"
MAX_HOP    = 3

# ── Active module prefixes — single source of truth ────────────────────────────
# Pulled directly from TABLE_PREFIX in .env — no meta_loader, no chatbot
# query/registry table, no extra system-table prefixes spliced in. Whatever
# is listed in TABLE_PREFIX is exactly what gets scanned, nothing more.
def get_module_prefixes() -> List[str]:
    raw = os.getenv("TABLE_PREFIX", "")
    prefixes = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not prefixes:
        log.warning("[config] TABLE_PREFIX is empty/unset in .env — no modules to process.")
    return prefixes


TABLE_PREFIXES: List[str] = get_module_prefixes()

# Columns so common they appear in almost every table — matching on these
# would create thousands of spurious edges and must be excluded everywhere.
_SKIP_COLS = {
    "tenant_id",
    "plant_id",
    "created_by",
    "updated_by",
    "created_at",
    "updated_at",
    "is_deleted",
    "is_active",
}


# ─────────────────────────────────────────────────────────────────────────────
# Query builders — parameterised by prefix
#
# Pattern logic:
#   Tables like MFC_M_BOM  → LIKE 'MFC[_]%'
#   Tables like sysdiagrams (no underscore) → LIKE 'sysdiagrams'  (exact)
#   Both cases are OR-combined so the filter is always correct.
# ─────────────────────────────────────────────────────────────────────────────

def _any_prefix_pattern(alias: str, prefixes: List[str]) -> str:
    """
    Return a SQL OR-clause fragment matching `alias.name` against ANY of the
    given prefixes — e.g. alias belongs to PRD_% OR SEQ_% OR MFC_% ... .

    This replaces the old single-prefix `_like`/`_like2` helpers. The old
    versions scoped from_pat and to_pat to the SAME single prefix, which
    made cross-module edges (e.g. PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER)
    structurally impossible to discover no matter which prefixes were
    configured. Using "any active module" on both sides independently means
    the from-side and to-side no longer have to be the same module.
    """
    clauses = []
    for prefix in prefixes:
        safe = prefix.replace("[", "[[]").replace("%", "[%]").replace("_", "[_]")
        clauses.append(f"{alias}.name LIKE '{safe}[_]%' OR {alias}.name = '{prefix}'")
    return "(" + " OR ".join(clauses) + ")"


def _q_enforced_fk(prefixes: List[str]) -> str:
    from_pat = _any_prefix_pattern("tp", prefixes)
    to_pat   = _any_prefix_pattern("tr", prefixes)
    return f"""
SELECT
    fk.name          AS fk_name,
    tp.name          AS from_table,
    cp.name          AS from_column,
    tr.name          AS to_table,
    cr.name          AS to_column,
    'enforced_fk'    AS edge_source
FROM sys.foreign_keys fk
INNER JOIN sys.foreign_key_columns fkc
    ON  fk.object_id = fkc.constraint_object_id
INNER JOIN sys.tables  tp ON fkc.parent_object_id      = tp.object_id
INNER JOIN sys.columns cp ON fkc.parent_object_id      = cp.object_id
                          AND fkc.parent_column_id     = cp.column_id
INNER JOIN sys.tables  tr ON fkc.referenced_object_id  = tr.object_id
INNER JOIN sys.columns cr ON fkc.referenced_object_id  = cr.object_id
                          AND fkc.referenced_column_id = cr.column_id
WHERE {from_pat}
  AND {to_pat}
ORDER BY tp.name, tr.name
"""


def _q_implicit_fk_via_pk(prefixes: List[str]) -> str:
    """Exact column-name match against the target table's primary key."""
    from_pat = _any_prefix_pattern("t_from", prefixes)
    to_pat   = _any_prefix_pattern("t_to", prefixes)
    return f"""
SELECT
    t_from.name                          AS from_table,
    c_from.name                          AS from_column,
    t_to.name                            AS to_table,
    c_to.name                            AS to_column,
    'implicit_pk_name_match'             AS edge_source
FROM sys.columns       c_from
JOIN sys.tables        t_from ON c_from.object_id  = t_from.object_id
JOIN sys.columns       c_to   ON c_to.name         = c_from.name
JOIN sys.tables        t_to   ON c_to.object_id    = t_to.object_id
JOIN sys.indexes       pk     ON pk.object_id      = t_to.object_id
                              AND pk.is_primary_key = 1
JOIN sys.index_columns ic     ON ic.object_id      = pk.object_id
                              AND ic.index_id       = pk.index_id
                              AND ic.column_id      = c_to.column_id
WHERE {from_pat}
  AND {to_pat}
  AND t_from.name <> t_to.name
  AND NOT EXISTS (
      SELECT 1
      FROM sys.indexes       pk2
      JOIN sys.index_columns ic2
          ON ic2.object_id = pk2.object_id
         AND ic2.index_id  = pk2.index_id
         AND ic2.column_id = c_from.column_id
      WHERE pk2.object_id      = t_from.object_id
        AND pk2.is_primary_key = 1
  )
ORDER BY t_from.name, c_from.name
"""


def _q_implicit_fk_via_pk_suffix(prefixes: List[str]) -> str:
    """
    Suffix column-name match against the target table's primary key.

    Catches cases the exact-match query misses, e.g. a child column named
    'INT_Item_ID' referencing MFC_M_ITEM.Item_ID — the child column ends
    with the parent PK column name but isn't an exact match.

    Exact matches are excluded here (they're already covered by
    implicit_pk_name_match) so the two sources never produce duplicate edges.
    """
    from_pat = _any_prefix_pattern("t_from", prefixes)
    to_pat   = _any_prefix_pattern("t_to", prefixes)
    return f"""
SELECT
    t_from.name                          AS from_table,
    c_from.name                          AS from_column,
    t_to.name                            AS to_table,
    c_to.name                            AS to_column,
    'implicit_pk_suffix_match'           AS edge_source
FROM sys.columns       c_from
JOIN sys.tables        t_from ON c_from.object_id  = t_from.object_id
JOIN sys.tables        t_to   ON {to_pat}
JOIN sys.indexes       pk     ON pk.object_id      = t_to.object_id
                              AND pk.is_primary_key = 1
JOIN sys.index_columns ic     ON ic.object_id      = pk.object_id
                              AND ic.index_id       = pk.index_id
JOIN sys.columns       c_to   ON c_to.object_id    = t_to.object_id
                              AND c_to.column_id    = ic.column_id
WHERE {from_pat}
  AND t_from.name <> t_to.name
  AND c_from.name <> c_to.name
  AND c_from.name LIKE '%\\_' + c_to.name ESCAPE '\\'
  AND NOT EXISTS (
      SELECT 1
      FROM sys.indexes       pk2
      JOIN sys.index_columns ic2
          ON ic2.object_id = pk2.object_id
         AND ic2.index_id  = pk2.index_id
         AND ic2.column_id = c_from.column_id
      WHERE pk2.object_id      = t_from.object_id
        AND pk2.is_primary_key = 1
  )
ORDER BY t_from.name, c_from.name
"""


# ─────────────────────────────────────────────────────────────────────────────
# DB connection
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection():
    try:
        from config.db import get_db_connection
        log.info("Using config.db.get_db_connection")
        return get_db_connection()
    except Exception as e:
        log.warning("config.db import failed (%s) — falling back to env vars", e)

    import pyodbc

    server   = os.environ.get("DB_SERVER",   "")
    database = os.environ.get("DB_NAME",     "")
    user     = os.environ.get("DB_USER",     "")
    password = os.environ.get("DB_PASSWORD", "")
    driver   = os.environ.get("DB_DRIVER",   "ODBC Driver 17 for SQL Server")

    if not server or not database:
        log.error("DB_SERVER and DB_NAME env vars are required when config.db is unavailable.")
        sys.exit(1)

    if user and password:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={database};"
            f"UID={user};PWD={password};"
            "TrustServerCertificate=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

    log.info("Connecting to %s / %s", server, database)
    return pyodbc.connect(conn_str)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — fetch all FK edges for a given prefix
# ─────────────────────────────────────────────────────────────────────────────

def _run_query(cur, sql: str, label: str) -> List[dict]:
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for r in rows:
        if "fk_name" not in r:
            r["fk_name"] = f"{r['edge_source']}__{r['from_table']}__{r['from_column']}"
    log.info("  [%s] %d edge(s)", label, len(rows))
    return rows


def fetch_all_edges(conn, prefixes: List[str]) -> List[dict]:
    """
    Fetch every FK/implicit-FK edge where BOTH the from-table and the
    to-table belong to ANY of the active prefixes — not necessarily the
    SAME prefix. This is what makes cross-module edges (e.g.
    PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER) discoverable at all.
    """
    cur = conn.cursor()

    enforced        = _run_query(cur, _q_enforced_fk(prefixes),               "enforced_fk")
    implicit_pk     = _run_query(cur, _q_implicit_fk_via_pk(prefixes),        "implicit_pk")
    implicit_suffix = _run_query(cur, _q_implicit_fk_via_pk_suffix(prefixes), "implicit_pk_suffix")

    seen: set = set()
    merged: List[dict] = []

    for edge in enforced + implicit_pk + implicit_suffix:
        if edge["from_column"].lower() in _SKIP_COLS:
            continue
        if edge["to_column"].lower() in _SKIP_COLS:
            continue
        key = (
            edge["from_table"].upper(),
            edge["from_column"].upper(),
            edge["to_table"].upper(),
            edge["to_column"].upper(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(edge)

    log.info(
        "  Total after merge+dedup: %d  (enforced=%d  implicit_pk=%d  implicit_pk_suffix=%d)",
        len(merged), len(enforced), len(implicit_pk), len(implicit_suffix),
    )
    return merged


def _owning_module(table_name: str, prefixes: List[str]) -> str:
    """
    Return which active prefix `table_name` belongs to, longest-prefix-first
    so e.g. a hypothetical 'SEQL_...' table doesn't false-match a shorter
    'SEQ' prefix. Falls back to the text before the first underscore.
    """
    upper = table_name.upper()
    for prefix in sorted(prefixes, key=len, reverse=True):
        p = prefix.upper()
        if upper == p or upper.startswith(p + "_"):
            return prefix
    return table_name.split("_", 1)[0] if "_" in table_name else table_name


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — build bidirectional adjacency graph
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(edges: List[dict]) -> Dict[str, List[dict]]:
    seen: set = set()
    adj: Dict[str, List[dict]] = defaultdict(list)

    for e in edges:
        ft = e["from_table"]
        fc = e["from_column"]
        tt = e["to_table"]
        tc = e["to_column"]
        fk = e.get("fk_name", "")
        es = e.get("edge_source", "")

        key = (ft.upper(), fc.upper(), tt.upper(), tc.upper())
        if key in seen:
            continue
        seen.add(key)

        adj[ft].append({
            "from_table":  ft, "from_column": fc,
            "to_table":    tt, "to_column":   tc,
            "fk_name":     fk, "edge_source": es,
        })

        rev_key = (tt.upper(), tc.upper(), ft.upper(), fc.upper())
        if rev_key not in seen:
            seen.add(rev_key)
            adj[tt].append({
                "from_table":  tt, "from_column": tc,
                "to_table":    ft, "to_column":   fc,
                "fk_name":     fk + "__REV",
                "edge_source": es + "__REV",
            })

    log.info(
        "  Adjacency: %d nodes, %d directed edges (incl. reverse)",
        len(adj), sum(len(v) for v in adj.values()),
    )
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — BFS: shortest path between every reachable (source, target) pair
# ─────────────────────────────────────────────────────────────────────────────

def find_all_paths(
    adj: Dict[str, List[dict]],
    max_hop: int = MAX_HOP,
) -> Dict[str, Dict[str, List[dict]]]:
    all_tables: set = set(adj.keys())
    for edges in adj.values():
        for e in edges:
            all_tables.add(e["to_table"])

    paths: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)

    for source in all_tables:
        queue: deque = deque()
        queue.append((source, []))
        visited = {source}

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hop:
                continue
            for edge in adj.get(node, []):
                neighbor = edge["to_table"]
                if neighbor in visited:
                    continue
                new_path = path + [edge]
                if neighbor not in paths[source]:
                    paths[source][neighbor] = new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    total = sum(len(v) for v in paths.values())
    log.info("  BFS: %d (source, target) pairs reachable (max %d hops)", total, max_hop)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — strip internal fields from paths
# ─────────────────────────────────────────────────────────────────────────────

def build_join_paths(
    paths: Dict[str, Dict[str, List[dict]]]
) -> Dict[str, Dict[str, List[dict]]]:
    join_paths: Dict[str, Dict[str, List[dict]]] = {}
    for source, targets in paths.items():
        join_paths[source] = {}
        for target, path in targets.items():
            join_paths[source][target] = [
                {
                    "from_table":  step["from_table"],
                    "from_column": step["from_column"],
                    "to_table":    step["to_table"],
                    "to_column":   step["to_column"],
                }
                for step in path
            ]
    return join_paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — write output JSON for one prefix
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_node(adj: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    return {table: sorted({e["to_table"] for e in edges}) for table, edges in adj.items()}


def write_json(
    module: str,
    edges: List[dict],
    adj: Dict[str, List[dict]],
    join_paths: Dict[str, Dict[str, List[dict]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hop_counts = {1: 0, 2: 0, 3: 0}
    cross_module_paths = 0
    for source, targets in join_paths.items():
        source_module = _owning_module(source, TABLE_PREFIXES)
        for target, path in targets.items():
            hops = len(path)
            if hops in hop_counts:
                hop_counts[hops] += 1
            if any(_owning_module(step["to_table"], TABLE_PREFIXES) != source_module for step in path):
                cross_module_paths += 1

    source_counts: Dict[str, int] = defaultdict(int)
    for e in edges:
        source_counts[e.get("edge_source", "unknown")] += 1

    payload = {
        "_meta": {
            "module":               module,
            "generated_at":         datetime.utcnow().isoformat() + "Z",
            "source":               "live DB introspection (sys.foreign_keys + sys.columns/indexes)",
            "max_hop":              MAX_HOP,
            "total_fk_edges":       len(edges),
            "edges_by_source":      dict(source_counts),
            "total_tables":         len(adj),
            "total_valid_paths":    sum(len(v) for v in join_paths.values()),
            "paths_by_hop":         hop_counts,
            "cross_module_paths":   cross_module_paths,
        },
        "graph":      build_graph_node(adj),
        "join_paths": join_paths,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info(
        "  Written → %s  (%d tables | %d paths [%d cross-module] | 1-hop:%d 2-hop:%d 3-hop:%d)",
        output_path, len(adj),
        sum(len(v) for v in join_paths.values()), cross_module_paths,
        hop_counts[1], hop_counts[2], hop_counts[3],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global pipeline — fetch once across ALL active modules, one adjacency/BFS
# pass over the combined graph (so multi-hop paths can cross module
# boundaries), then split the result into one output file per module.
# ─────────────────────────────────────────────────────────────────────────────

def process_all(conn, prefixes: List[str]) -> List[str]:
    """
    Run the full edge-discovery → adjacency → BFS → per-module-write
    pipeline ONCE across every active module combined. Returns the list of
    modules that got a non-empty output file.

    This replaces the old per-prefix loop, where from_pat and to_pat were
    both scoped to the SAME single prefix — that made a cross-module edge
    like PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER structurally undiscoverable
    no matter which prefixes were configured, since PRD's run excluded SEQ
    on the to-side and SEQ's run excluded PRD on the from-side.
    """
    log.info("── GLOBAL FETCH across %d modules: %s ──", len(prefixes), prefixes)

    edges = fetch_all_edges(conn, prefixes)
    if not edges:
        log.warning("  No edges found across any active module.")
        return []

    same_module  = sum(
        1 for e in edges
        if _owning_module(e["from_table"], prefixes) == _owning_module(e["to_table"], prefixes)
    )
    cross_module = len(edges) - same_module
    log.info(
        "  Total edges: %d  (same-module=%d, cross-module=%d)",
        len(edges), same_module, cross_module,
    )
    if cross_module:
        for e in edges:
            if _owning_module(e["from_table"], prefixes) != _owning_module(e["to_table"], prefixes):
                log.info(
                    "  [cross-module] %s.%s → %s.%s",
                    e["from_table"], e["from_column"], e["to_table"], e["to_column"],
                )

    adj        = build_adjacency(edges)
    raw_paths  = find_all_paths(adj, max_hop=MAX_HOP)
    join_paths = build_join_paths(raw_paths)

    # Write ONE combined file across all active modules — no per-module
    # split. All tables, edges, and join paths (including cross-module
    # hops) go into a single graph under a single "PROD" label.
    combined_label = "PROD"
    filename    = "prod_join_graph.json"
    output_path = OUTPUT_DIR / filename
    write_json(combined_label, edges, adj, join_paths, output_path)

    return prefixes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Join Graph Builder — %d active modules ===", len(TABLE_PREFIXES))
    log.info("Modules: %s", TABLE_PREFIXES)
    log.info("Output dir: %s", OUTPUT_DIR)

    with _get_connection() as conn:
        written = process_all(conn, TABLE_PREFIXES)

    log.info("")
    log.info("=== Summary ===")
    if written:
        log.info("Modules combined into prod_join_graph.json (%d): %s", len(written), ", ".join(written))
    else:
        log.info("No edges found — prod_join_graph.json not written.")
    log.info("Done. Re-run graph_loader.py to push updated graphs into Neo4j.")


if __name__ == "__main__":
    main()