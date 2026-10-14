from __future__ import annotations
import logging
import re
from typing import Dict, List, Optional, Tuple

import os
log = logging.getLogger("NL2SQL")

_PROBE_TOP = 1

def _load_master_prefixes() -> Tuple[str, ...]:
    """
    Build master/lookup table prefix list from env vars.

    Priority order:
      1. TABLE_NAME_PREFIXES  — explicit full prefixes e.g. "MFC_M_,MFC_L_,MFC_T_"
         Only _M_ and _L_ entries are used (transactional _T_ tables are not masters).
      2. TABLE_PREFIX         — short module codes e.g. "MFC,ALM,USR"
         Expanded to "<CODE>_M_" and "<CODE>_L_" automatically.
      3. Hard fallback        — ("MFC_M_", "MFC_L_") so the app still works with
         no env configured.
    """
    # Strategy 1 — TABLE_NAME_PREFIXES (filter to _M_ and _L_ only)
    raw_name = os.getenv("TABLE_NAME_PREFIXES", "").strip()
    if raw_name:
        prefixes = tuple(
            p.strip() for p in raw_name.split(",")
            if p.strip() and (p.strip().upper().endswith("_M_") or p.strip().upper().endswith("_L_"))
        )
        if prefixes:
            return prefixes

    # Strategy 2 — TABLE_PREFIX short codes
    raw_code = os.getenv("TABLE_PREFIX", "").strip()
    if raw_code:
        codes = [c.strip().upper() for c in raw_code.split(",") if c.strip()]
        prefixes = tuple(f"{c}_M_" for c in codes) + tuple(f"{c}_L_" for c in codes)
        if prefixes:
            return prefixes

    # Strategy 3 — fallback
    return ("MFC_M_", "MFC_L_")

_DEFAULT_MASTER_PREFIXES: Tuple[str, ...] = _load_master_prefixes()

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_text_columns(conn, tables: List[str]) -> Dict[str, List[str]]:
    """
    One INFORMATION_SCHEMA round-trip for all tables in `tables`.
    Returns {table_name: [varchar_col, …]} — only text-type columns.
    """
    if not tables:
        return {}
    placeholders = ", ".join("?" * len(tables))
    sql = (
        "SELECT TABLE_NAME, COLUMN_NAME "
        "FROM   INFORMATION_SCHEMA.COLUMNS "
        f"WHERE  TABLE_NAME IN ({placeholders}) "
        "  AND  DATA_TYPE  IN ('varchar','nvarchar','char','nchar','text','ntext') "
        "ORDER BY TABLE_NAME, ORDINAL_POSITION"
    )
    try:
        cur = conn.cursor()
        cur.execute(sql, tables)
        result: Dict[str, List[str]] = {}
        for tbl, col in cur.fetchall():
            result.setdefault(tbl, []).append(col)
        return result
    except Exception as exc:
        log.warning("[value_searcher] _fetch_text_columns failed: %s", exc)
        return {}


def _score_column(col: str, value: str) -> int:
    """Higher score = more likely to be the business-key column for `value`.

    `value` may arrive as upper-case (value_upper) — normalise before
    tokenising so alphanumeric boundaries in codes like 'WOCNC0102' are
    split correctly (letters vs digits) rather than treated as one token
    that never matches a column-name word.
    """
    _AUDIT_TOKENS  = {"cuser", "uuser", "notes", "remarks", "description", "comment"}
    _GOOD_SUFFIXES = ("_name", "_number", "_code", "_no", "_key", "_ref")

    col_lower  = col.lower()
    col_tokens = set(re.split(r"[_\s]+", col_lower))

    if col_tokens & _AUDIT_TOKENS:
        return -1
    _val_lower = value.lower()
    _raw_parts = re.split(r"[-_\s]+", _val_lower)
    val_tokens: set = set()
    for part in _raw_parts:
        
        sub = re.split(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", part)
        val_tokens.update(t for t in sub if len(t) >= 3)

    score = len(val_tokens & col_tokens) * 3
    if any(col_lower.endswith(s) for s in _GOOD_SUFFIXES):
        score += 2
    return score

def _probe_table(conn, table: str, text_cols: List[str],
                 value_upper: str) -> Optional[Tuple[str, str]]:
    """
    Try each text column in `table` for an exact case-insensitive match.
    Columns are sorted by score so the best candidate is tried first.
    Returns (table, col) on first hit, None if no match.
    """
    if not text_cols:
        return None

    ordered = sorted(text_cols,
                     key=lambda c: _score_column(c, value_upper),
                     reverse=True)
    cur = conn.cursor()
    for col in ordered:
        try:
            cur.execute(
                f"SELECT TOP {_PROBE_TOP} [{col}] "
                f"FROM   [{table}] WITH (NOLOCK) "
                f"WHERE  UPPER([{col}]) = ?",
                (value_upper,),
            )
            if cur.fetchone() is not None:
                log.info("[value_searcher] ✅ %s.%s = %r", table, col, value_upper)
                return (table, col)
        except Exception as exc:
            log.debug("[value_searcher] skip %s.%s: %s", table, col, exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def search_value_in_db(
    conn,
    value: str,
    candidate_tables: List[str],
    all_known_tables: List[str],
    master_prefixes: Tuple[str, ...] = _DEFAULT_MASTER_PREFIXES,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find which DB table+column stores `value`.

    The pipeline calls this function TWICE — once per phase — so each call
    is responsible for exactly one phase:

    Phase 1 (Step 2 in pipeline):
        candidate_tables = BM25 semantic tables
        all_known_tables = []           ← pass empty to skip Phase 2 here
        Returns on first hit, (None, None) on miss.

    Phase 2 (Step 3 in pipeline):
        candidate_tables = []           ← already checked, skip
        all_known_tables = full META table list
        Scans in TWO sub-passes so master/link tables are tried first
        (cheap, high-precision business keys) but transactional (_T_)
        tables are never permanently excluded:
          2a — tables starting with a master/link prefix (e.g. MFC_M_, MFC_L_)
          2b — fallback: every remaining table in all_known_tables that 2a
               didn't already cover (this includes _T_ transactional tables,
               where WO/order-style literals like RWO984 actually live).
        Returns on first hit (from whichever sub-pass hits first), (None, None)
        if both sub-passes miss.

    Parameters
    ----------
    conn              : open DB connection (caller owns context manager)
    value             : raw business-key string from the question
    candidate_tables  : tables to probe in Phase 1 (pass [] to skip)
    all_known_tables  : full META table list for Phase-2 expansion
                        (pass [] to skip Phase 2)
    master_prefixes   : Phase-2a scans tables starting with one of these first;
                        Phase-2b then falls back to the rest of
                        all_known_tables (so _T_ tables are still reachable
                        even if excluded from master_prefixes upstream)

    Returns
    -------
    (owner_table, owner_col)  or  (None, None)
    """
    value_upper = value.strip().upper()
    if not value_upper:
        return None, None

    # ── Phase 1: probe candidate (semantic) tables ────────────────────────────
    if candidate_tables:
        log.info(
            "[value_searcher] Phase-1 — probing %d candidate table(s): %s",
            len(candidate_tables), candidate_tables,
        )
        p1_cols = _fetch_text_columns(conn, candidate_tables)
        for tbl in candidate_tables:
            hit = _probe_table(conn, tbl, p1_cols.get(tbl, []), value_upper)
            if hit:
                log.info("[value_searcher] Phase-1 ✅ found in %s.%s", hit[0], hit[1])
                return hit
        log.info(
            "[value_searcher] Phase-1 miss — %r not in %d candidate table(s)",
            value_upper, len(candidate_tables),
        )

    # ── Phase 2: master + lookup tables (business keys live here) ────────────
    if not all_known_tables:
        # Caller passed empty list — Phase 2 intentionally skipped for this call
        return None, None

    already_checked = {t.lower() for t in candidate_tables}

    # ── Phase 2a: master/link tables first (cheap, high-precision) ───────────
    p2a_tables = [
        t for t in all_known_tables
        if t.lower() not in already_checked
        and any(t.upper().startswith(pfx.upper()) for pfx in master_prefixes)
    ]

    if p2a_tables:
        log.info(
            "[value_searcher] Phase-2a — probing %d master/link table(s)",
            len(p2a_tables),
        )
        p2a_cols = _fetch_text_columns(conn, p2a_tables)
        for tbl in p2a_tables:
            hit = _probe_table(conn, tbl, p2a_cols.get(tbl, []), value_upper)
            if hit:
                log.info("[value_searcher] Phase-2a ✅ found in %s.%s", hit[0], hit[1])
                return hit
        log.info(
            "[value_searcher] Phase-2a miss — %r not found in %d master/link table(s)",
            value_upper, len(p2a_tables),
        )
    else:
        log.info(
            "[value_searcher] Phase-2a — no master/link tables found "
            "(prefixes=%s), skipping straight to Phase-2b.",
            master_prefixes,
        )

    # ── Phase 2b: fallback — everything else in all_known_tables ─────────────
    # Covers transactional (_T_) tables and any other table not caught by
    # master_prefixes. This is where WO/order-style literals (e.g. RWO984)
    # actually live, and it must never be permanently excluded just because
    # the caller narrowed master_prefixes to _M_/_L_ only.
    already_checked_2a = already_checked | {t.lower() for t in p2a_tables}
    p2b_tables = [
        t for t in all_known_tables
        if t.lower() not in already_checked_2a
    ]

    if not p2b_tables:
        log.info("[value_searcher] Phase-2b — no remaining tables to scan. Returning miss.")
        return None, None

    log.info(
        "[value_searcher] Phase-2b — probing %d remaining table(s) "
        "(transactional + other)",
        len(p2b_tables),
    )
    p2b_cols = _fetch_text_columns(conn, p2b_tables)

    for tbl in p2b_tables:
        hit = _probe_table(conn, tbl, p2b_cols.get(tbl, []), value_upper)
        if hit:
            log.info("[value_searcher] Phase-2b ✅ found in %s.%s", hit[0], hit[1])
            return hit

    log.info(
        "[value_searcher] Phase-2b miss — %r not found in %d remaining table(s)",
        value_upper, len(p2b_tables),
    )
    return None, None