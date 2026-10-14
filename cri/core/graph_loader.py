from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("graph_loader")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── locate join_graphs folder ─────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent          # core/
_PROJECT     = _HERE.parent                             # project root
_JOIN_GRAPHS = _PROJECT / "data" / "join_graphs"        # data/join_graphs/<module>_join_graph.json

# ── Neo4j connection (reads from .env / env vars) ─────────────────────────────
_NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "cncdb0804")


def _get_driver():
    """Return a neo4j.GraphDatabase driver or raise with a clear message."""
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        raise RuntimeError(
            "neo4j driver not installed. Run:  pip install neo4j"
        )
    return GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))


# ── cardinality inference ─────────────────────────────────────────────────────
def _infer_cardinality(from_col: str, to_col: str) -> str:
    """
    Assign a join cost weight from the column names alone.
      - Both PKs (col named *_ID, same name both sides)  → one_to_one   cost 1
      - FK → PK (from is a FK, to is the PK)             → many_to_one  cost 1
      - PK → FK (from is PK, to is a FK)                 → one_to_many  cost 1
      - anything involving Alternative/Suc/Pre cols       → many_to_many cost 3
    """
    fc, tc = from_col.upper(), to_col.upper()
    ambiguous_markers = ("ALTERNATIVE", "PRE_", "SUC_", "PREV", "NEXT")
    if any(m in fc or m in tc for m in ambiguous_markers):
        return "many_to_many"
    # Exact match, or a prefixed/suffixed variant of the same column
    # (e.g. INT_Item_ID → Item_ID from implicit_pk_suffix_match edges)
    if fc == tc or fc.endswith("_" + tc) or tc.endswith("_" + fc):
        return "many_to_one"
    return "one_to_many"


_CARDINALITY_COST = {
    "one_to_one":   1,
    "many_to_one":  1,
    "one_to_many":  1,
    "many_to_many": 3,
}


def _module_from_path(path: Path) -> str:
    """
    Derive module prefix from filename.
    e.g.  com_join_graph.json  →  COM
          mfc_join_graph.json  →  MFC
          wms_join_graph.json  →  WMS
    """
    return path.name.split("_join_graph")[0].upper()


# ─────────────────────────────────────────────────────────────────────────────
def load_graph(driver, join_graph_path: Path, module: str) -> None:
    """
    Reads prod_join_graph.json and writes ALL edges into Neo4j — including
    cross-module hops (e.g. PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER).

    Nodes  : (:Table {name, module})
    Edges  : [:JOINS_TO {from_col, to_col, cardinality, cost, module}]

    The file is now a single combined graph across every prefix in
    TABLE_PREFIX (MFC, PRD, SEQ, ...), built by build_join_graph.py's global
    fetch. There is no per-table-prefix module to enforce anymore — the
    intra-module-only filter that used to live here would drop every
    cross-module edge (nothing is actually named "PROD_*"), which is exactly
    the traversal you need working. So: load every edge in the file as-is,
    tagged under module="PROD" as one combined graph namespace.
    """
    data       = json.loads(join_graph_path.read_text(encoding="utf-8"))
    join_paths: dict = data.get("join_paths", {})

    # ── collect unique 1-hop edges (any prefix, cross-module included) ───────
    direct_edges: dict = {}   # (from_table, from_col, to_table, to_col) → True
    all_tables:   set  = set()

    for source, targets in join_paths.items():
        for target, steps in targets.items():
            for step in steps:
                ft = step.get("from_table",  "").upper()
                fc = step.get("from_column", "")
                tt = step.get("to_table",    "").upper()
                tc = step.get("to_column",   "")

                if not (ft and fc and tt and tc):
                    continue

                all_tables.add(ft)
                all_tables.add(tt)
                direct_edges[(ft, fc, tt, tc)] = True

    all_tables.discard("")
    log.info(
        "[%s] Parsed %d unique 1-hop edges across %d tables (cross-module included)",
        module, len(direct_edges), len(all_tables),
    )

    with driver.session() as session:

        # ── wipe THIS module only ─────────────────────────────────────────────
        session.run(
            "MATCH ()-[r:JOINS_TO {module: $mod}]->() DELETE r",
            mod=module,
        )
        session.run(
            "MATCH (t:Table {module: $mod}) DETACH DELETE t",
            mod=module,
        )
        log.info("[%s] Existing nodes/edges cleared.", module)

        # ── create Table nodes (with module tag) ──────────────────────────────
        for name in sorted(all_tables):
            session.run(
                "MERGE (:Table {name: $name, module: $mod})",
                name=name, mod=module,
            )
        log.info("[%s] Created %d Table nodes.", module, len(all_tables))

        # ── create JOINS_TO edges (intra-module, 1-hop only) ──────────────────
        edge_count = 0
        for (from_table, from_col, to_table, to_col) in direct_edges:
            cardinality = _infer_cardinality(from_col, to_col)
            cost        = _CARDINALITY_COST[cardinality]
            session.run(
                """
                MATCH (a:Table {name: $from_t, module: $mod}),
                      (b:Table {name: $to_t,   module: $mod})
                MERGE (a)-[r:JOINS_TO {from_col: $fc, to_col: $tc, module: $mod}]->(b)
                SET r.cardinality = $card,
                    r.cost        = $cost
                """,
                from_t = from_table,
                to_t   = to_table,
                fc     = from_col,
                tc     = to_col,
                mod    = module,
                card   = cardinality,
                cost   = cost,
            )
            edge_count += 1

        log.info("[%s] Created %d JOINS_TO relationships.", module, edge_count)


# ─────────────────────────────────────────────────────────────────────────────
def discover_join_graphs(folder: Path = _JOIN_GRAPHS) -> list[Path]:
    """
    build_join_graph.py writes ONE combined file — prod_join_graph.json —
    across every prefix in TABLE_PREFIX (MFC, PRD, SEQ, ...), so that
    multi-hop paths can cross module boundaries. There's nothing left to
    filter by TABLE_PREFIX here (the old per-prefix files
    mfc_join_graph.json / prd_join_graph.json / seq_join_graph.json are no
    longer written) — just load the single combined file.
    """
    if not folder.exists():
        log.error("join_graphs folder not found: %s", folder)
        return []

    target = folder / "prod_join_graph.json"
    if not target.exists():
        log.error(
            "%s not found — run build_join_graph.py first.", target.name
        )
        return []

    log.info("Discovered combined join graph: %s", target.name)
    return [target]


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info(f"Connecting to Neo4j at {_NEO4J_URI} …")
    driver = _get_driver()
    try:
        driver.verify_connectivity()
        log.info("Connection OK.")
    except Exception as exc:
        log.error(f"Cannot connect to Neo4j: {exc}")
        log.error("\n👉 Neo4j not running? See NEO4J_SETUP.md for installation steps.")
        sys.exit(1)

    files = discover_join_graphs()
    if not files:
        log.error("No *_join_graph.json files found under join_graphs/. Exiting.")
        driver.close()
        sys.exit(1)

    # ── one-time schema setup ─────────────────────────────────────────────────
    with driver.session() as session:
        # Purge any legacy Table nodes that lack a module property (orphans from
        # old schema runs) — these would block the NODE KEY constraint below.
        result = session.run(
            "MATCH (t:Table) WHERE t.module IS NULL "
            "WITH count(t) AS n RETURN n"
        )
        orphan_count = result.single()["n"]
        if orphan_count:
            session.run("MATCH (t:Table) WHERE t.module IS NULL DETACH DELETE t")
            log.warning("Purged %d orphan Table node(s) with no module property.", orphan_count)

        # Drop old single-property constraint if still present from a prior run.
        session.run("DROP CONSTRAINT table_name_unique IF EXISTS")

        # Composite NODE KEY: (name, module) — allows same table name across modules.
        session.run(
            "CREATE CONSTRAINT table_name_module_unique IF NOT EXISTS "
            "FOR (t:Table) REQUIRE (t.name, t.module) IS NODE KEY"
        )
        log.info("Schema constraint ready: (Table.name, Table.module) IS NODE KEY")

    for path in files:
        module = _module_from_path(path)
        log.info("=" * 55)
        log.info("Loading module: %s  ←  %s", module, path.name)
        log.info("=" * 55)
        try:
            load_graph(driver, path, module)
            log.info("[%s] ✅ Done.", module)
        except Exception as exc:
            log.error("[%s] ❌ Failed: %s", module, exc)

    driver.close()
    log.info("All modules loaded. You can now start the chatbot server.")


if __name__ == "__main__":
    main()