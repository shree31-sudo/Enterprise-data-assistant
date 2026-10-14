from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

log = logging.getLogger("neo4j_helper")

# ─────────────────────────────────────────────────────────────
# Config  (resolved at import time from env; no I/O yet)
# ─────────────────────────────────────────────────────────────
_URI:  str = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_USER: str = os.getenv("NEO4J_USER",     "neo4j")
_PASS: str = os.getenv("NEO4J_PASSWORD", "")

# Maximum number of (table, hops) pairs held in the in-process cache.
_MAX_CACHE_SIZE: int = int(os.getenv("NEO4J_CACHE_MAX_SIZE", "5000"))

# ─────────────────────────────────────────────────────────────
# Lazy driver — created on first use, never at import time
# ─────────────────────────────────────────────────────────────
_driver_lock  = threading.Lock()
_driver       = None   
_driver_ok: bool = False   


def _get_driver():
    """Return the shared driver, creating it on first call."""
    global _driver, _driver_ok
    if _driver is not None:
        return _driver
    with _driver_lock:
        if _driver is not None:        
            return _driver
        try:
            from neo4j import GraphDatabase 
            _driver = GraphDatabase.driver(_URI, auth=(_USER, _PASS))
            _driver.verify_connectivity()
            _driver_ok = True
            log.info("[neo4j_helper] Driver connected: %s (user=%s)", _URI, _USER)
        except Exception as exc:
            _driver_ok = False
            log.warning(
                "[neo4j_helper] Could not connect to Neo4j at %s — "
                "neighbour lookups will return [] until it becomes reachable. (%s)",
                _URI, exc,
            )
    return _driver


def close() -> None:
    """Close the shared driver and release the connection pool.

    Call this from the FastAPI lifespan shutdown handler, e.g.:
        from core import neo4j_helper
        neo4j_helper.close()
    """
    global _driver, _driver_ok
    with _driver_lock:
        if _driver is not None:
            try:
                _driver.close()
                log.info("[neo4j_helper] Driver closed.")
            except Exception as exc:
                log.warning("[neo4j_helper] Error closing driver: %s", exc)
            finally:
                _driver    = None
                _driver_ok = False


# ─────────────────────────────────────────────────────────────
# In-memory neighbour cache  (LRU-style eviction)
# ─────────────────────────────────────────────────────────────
_NEIGHBOR_CACHE: Dict[str, List[str]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_evict_quarter() -> None:
    """Drop the oldest 25 % of cache entries (called while _CACHE_LOCK is held)."""
    n_drop = max(1, len(_NEIGHBOR_CACHE) // 4)
    keys_to_drop = list(_NEIGHBOR_CACHE.keys())[:n_drop]
    for k in keys_to_drop:
        del _NEIGHBOR_CACHE[k]
    log.debug("[neo4j_helper] Cache evicted %d entries (LRU quarter-flush).", n_drop)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def get_neighbor_tables(table: str, hops: int = 2) -> List[str]:
    """Return FK-connected neighbour tables for *table* up to *hops* away.

    Parameters
    ----------
    table:
        Root table name (exact match against the ``name`` property on
        ``:Table`` nodes in Neo4j).
    hops:
        Traversal depth (clamped to 1–3).
        Recommended values:
            ``1`` — simple / single-table / time-based queries
            ``2`` — relational / join queries

    Returns
    -------
    Sorted list of neighbour table names, or ``[]`` on any error /
    unavailability (callers fall back to BFS over join_paths).

    Notes
    -----
    The Cypher query uses *outbound* FK edges only::

        (t)-[:FK_TO*1..N]->(n)

    This matches tables that *t* has a foreign key pointing at — i.e. genuine
    join partners — rather than every table in the neighbourhood including
    tables that merely reference *t*.
    """
    if not table:
        return []

    hops = max(1, min(hops, 3))
    cache_key = f"{table}:{hops}"

    # ── Cache hit ────────────────────────────────────────────────────────────
    with _CACHE_LOCK:
        cached = _NEIGHBOR_CACHE.get(cache_key)
        if cached is not None:
            return cached


    query = (
        f"MATCH (t:Table {{name:$table}})-[:JOINS_TO*1..{hops}]->(n:Table) "
        "RETURN DISTINCT n.name AS table"
    )

    try:
        drv = _get_driver()
        if drv is None:
            return []

        with drv.session() as session:
            rows = session.run(query, table=table)
            result: List[str] = sorted({
                r["table"]
                for r in rows
                if r.get("table") and r["table"] != table
            })

        # ── Cache store ──────────────────────────────────────────────────────
        with _CACHE_LOCK:
            if len(_NEIGHBOR_CACHE) >= _MAX_CACHE_SIZE:
                _cache_evict_quarter()
            _NEIGHBOR_CACHE[cache_key] = result

        log.debug(
            "[neo4j_helper] %r hops=%d → %d neighbour(s): %s",
            table, hops, len(result), result,
        )
        return result

    except Exception as exc:
        log.warning(
            "[neo4j_helper] Failed fetching neighbours for %r (hops=%d): %s",
            table, hops, exc,
        )
        return []


# ─────────────────────────────────────────────────────────────
# Cache utilities
# ─────────────────────────────────────────────────────────────

def clear_neighbor_cache() -> None:
    """Flush the entire in-process neighbour cache."""
    with _CACHE_LOCK:
        _NEIGHBOR_CACHE.clear()
    log.info("[neo4j_helper] Neighbour cache cleared.")


def get_neighbor_cache_stats() -> dict:
    """Return a snapshot of current cache occupancy."""
    with _CACHE_LOCK:
        return {
            "entries":   len(_NEIGHBOR_CACHE),
            "max_size":  _MAX_CACHE_SIZE,
            "keys_sample": list(_NEIGHBOR_CACHE.keys())[:10],
        }


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

def neo4j_healthcheck() -> bool:
    """Return True if Neo4j is reachable, False otherwise."""
    try:
        drv = _get_driver()
        if drv is None:
            return False
        drv.verify_connectivity()
        return True
    except Exception:
        return False
