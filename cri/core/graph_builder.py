from __future__ import annotations
import os, logging
from pathlib import Path

log = logging.getLogger("graph_builder")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from dotenv import load_dotenv
_PROJECT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT / ".env")

_NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "cncdb0804")

from config.db import get_db_connection


def get_module_prefixes() -> list[str]:
    """
    Active module prefixes come straight from TABLE_PREFIX in .env — no
    meta_loader / chatbot query-table lookup. This is the single source of
    truth for which modules the graph builder is allowed to touch.
    """
    raw = os.getenv("TABLE_PREFIX", "")
    prefixes = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not prefixes:
        log.warning("[config] TABLE_PREFIX is empty/unset in .env — no modules to process.")
    return prefixes


# ─────────────────────────────────────────────────────────────────────────────
# Cardinality
# ─────────────────────────────────────────────────────────────────────────────
_CARDINALITY_COST = {
    "one_to_one":   1,
    "many_to_one":  1,
    "one_to_many":  1,
    "many_to_many": 3,
}

def _infer_cardinality(from_col: str, to_col: str) -> str:
    fc, tc = from_col.upper(), to_col.upper()
    if any(m in fc or m in tc for m in ("ALTERNATIVE", "PRE_", "SUC_", "PREV", "NEXT")):
        return "many_to_many"
    if fc == tc:
        return "many_to_one"
    return "one_to_many"


def _table_module(table_name: str) -> str:
    """
    Derive a table's module label from its name prefix, e.g.
    'PRD_L_REWORK_ORDER' -> 'PRD', 'SEQ_L_WORK_ORDER' -> 'SEQ'.
    Falls back to the whole name if no underscore is present (shouldn't
    happen for any real table, but avoids a crash on unexpected input).
    """
    return table_name.split("_", 1)[0].upper() if "_" in table_name else table_name.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Fetch ALL FK relationships for the active modules, INCLUDING
# cross-module edges (e.g. PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER).
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_all_relationships(active_modules: list[str] | None = None) -> list[dict]:
    """
    Queries INFORMATION_SCHEMA directly. No JSON file, no manual filter beyond
    the active module list from module_tables.json / TABLE_PREFIX.

    Unlike the old MFC-only version, this does NOT require both sides of the
    FK to belong to the same module — a relationship where the FK table is
    PRD_% and the referenced table is SEQ_% (or any other active module) is
    captured too. This is what makes cross-module bridge tables (e.g. a
    shared work-order table referenced by PRD, QAL, etc.) discoverable by
    the join graph at all.
    """
    modules = active_modules or get_module_prefixes()
    if not modules:
        log.warning("[db] No active modules resolved from TABLE_PREFIX — nothing to fetch.")
        return []

    # Build a WHERE clause that admits a table if it belongs to ANY active
    # module, on EITHER side of the FK — this is what enables cross-module
    # edges instead of restricting both sides to the same prefix.
    like_clauses_fk = " OR ".join(f"fk.TABLE_NAME LIKE '{m}\\_%' ESCAPE '\\'" for m in modules)
    like_clauses_pk = " OR ".join(f"pk.TABLE_NAME LIKE '{m}\\_%' ESCAPE '\\'" for m in modules)

    query = f"""
        SELECT
            fk.TABLE_NAME   AS from_table,
            fk.COLUMN_NAME  AS from_col,
            pk.TABLE_NAME   AS to_table,
            pk.COLUMN_NAME  AS to_col
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk
            ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk
            ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
        WHERE ({like_clauses_fk})
          AND ({like_clauses_pk})
        ORDER BY fk.TABLE_NAME
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

    relationships = [
        {
            "from_table": r.from_table.upper(),
            "from_col":   r.from_col,
            "to_table":   r.to_table.upper(),
            "to_col":     r.to_col,
        }
        for r in rows
    ]

    same_module   = sum(1 for r in relationships if _table_module(r["from_table"]) == _table_module(r["to_table"]))
    cross_module  = len(relationships) - same_module

    log.info(
        f"[db] FK relationships fetched: {len(relationships)} "
        f"(same-module={same_module}, cross-module={cross_module}) "
        f"across modules={modules}"
    )

    for r in relationships:
        tag = "" if _table_module(r["from_table"]) == _table_module(r["to_table"]) else "  [cross-module]"
        log.info(
            f"  {r['from_table']}.{r['from_col']}"
            f" → {r['to_table']}.{r['to_col']}{tag}"
        )

    return relationships


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load into Neo4j
# ─────────────────────────────────────────────────────────────────────────────
def _load_into_neo4j(relationships: list[dict], active_modules: list[str]) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        _NEO4J_URI,
        auth=(_NEO4J_USER, _NEO4J_PASS)
    )
    driver.verify_connectivity()
    log.info("[neo4j] Connection OK.")

    with driver.session() as session:

        # Uniqueness constraint
        session.run(
            "CREATE CONSTRAINT table_name_unique IF NOT EXISTS "
            "FOR (t:Table) REQUIRE t.name IS UNIQUE"
        )

        # Wipe existing nodes for ALL active modules being rebuilt this run —
        # not just MFC. Each module's prefix is matched individually so we
        # never touch a module that isn't part of this run.
        for module in active_modules:
            session.run(
                "MATCH (t:Table) WHERE t.name STARTS WITH $prefix "
                "DETACH DELETE t",
                prefix=f"{module}_",
            )
        log.info(f"[neo4j] Existing nodes/edges cleared for modules: {active_modules}")

        # Create Table nodes — module label derived per-table, not hardcoded.
        all_tables = (
            {r["from_table"] for r in relationships} |
            {r["to_table"]   for r in relationships}
        )
        for name in sorted(all_tables):
            session.run(
                "MERGE (t:Table {name: $name}) SET t.module = $module",
                name=name,
                module=_table_module(name),
            )
        log.info(f"[neo4j] Created/updated {len(all_tables)} Table nodes.")

        # Create JOINS_TO edges
        for r in relationships:
            card = _infer_cardinality(r["from_col"], r["to_col"])
            cost = _CARDINALITY_COST[card]
            session.run("""
                MATCH (a:Table {name: $from_t}),
                      (b:Table {name: $to_t})
                MERGE (a)-[rel:JOINS_TO {from_col: $fc, to_col: $tc}]->(b)
                SET rel.cardinality = $card,
                    rel.cost        = $cost
            """,
                from_t = r["from_table"],
                to_t   = r["to_table"],
                fc     = r["from_col"],
                tc     = r["to_col"],
                card   = card,
                cost   = cost,
            )
        log.info(f"[neo4j] Created {len(relationships)} JOINS_TO edges.")

        # Final summary — per module, plus a cross-module total.
        summary = session.run("""
            MATCH (a:Table)-[r:JOINS_TO]->(b:Table)
            RETURN a.module AS module, count(r) AS edges, count(DISTINCT a) AS nodes
        """).data()
        for row in summary:
            log.info(
                f"[neo4j] Graph summary — module: {row['module']}, "
                f"nodes: {row['nodes']}, edges: {row['edges']}"
            )
        cross = session.run("""
            MATCH (a:Table)-[r:JOINS_TO]->(b:Table)
            WHERE a.module <> b.module
            RETURN count(r) AS cross_module_edges
        """).single()
        log.info(f"[neo4j] Cross-module edges: {cross['cross_module_edges']}")

    driver.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> None:
    active_modules = get_module_prefixes()

    log.info("=" * 55)
    log.info("Graph Builder — SQL Server → Neo4j")
    log.info(f"Active modules: {active_modules}")
    log.info("=" * 55)

    relationships = _fetch_all_relationships(active_modules)

    if not relationships:
        log.error(
            "No FK relationships found in SQL Server for the active modules. "
            "Check DB credentials in .env and module_tables.json / TABLE_PREFIX."
        )
        return

    _load_into_neo4j(relationships, active_modules)

    log.info("=" * 55)
    log.info("Done \u2705  Neo4j is now in sync with SQL Server.")
    log.info("=" * 55)


if __name__ == "__main__":
    build_graph()