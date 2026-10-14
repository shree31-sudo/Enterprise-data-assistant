

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

log = logging.getLogger("LOVResolver")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — ENV-DRIVEN CONFIG
# All DB references resolved from .env — never hardcoded.
# ══════════════════════════════════════════════════════════════════════════════

def _lov_table_ref(table_env_key: str, fallback: str) -> str:
    """
    Build a fully-qualified table reference from env vars.
    Examples:
        [LAMI_CNC_PROD0302].[dbo].[COM_L_LOV_DETAILS]
        [dbo].[COM_L_LOV_DETAILS]           (when LOV_DB_NAME not set)
        [COM_L_LOV_DETAILS]                 (when both LOV_DB_NAME and LOV_DB_SCHEMA not set)
    """
    db_name    = os.getenv("LOV_DB_NAME",   "").strip()
    db_schema  = os.getenv("LOV_DB_SCHEMA", "dbo").strip()
    table_name = os.getenv(table_env_key,   fallback).strip()

    if db_name and db_schema:
        return f"[{db_name}].[{db_schema}].[{table_name}]"
    if db_schema:
        return f"[{db_schema}].[{table_name}]"
    return f"[{table_name}]"


# Resolved once at import — changing env vars requires a server restart
_LOV_DETAILS_REF: str = _lov_table_ref("LOV_TABLE_DETAILS", "COM_L_LOV_DETAILS")
_LOV_MASTER_REF:  str = _lov_table_ref("LOV_TABLE_MASTER",  "COM_M_LOV")

log.info(
    f"[LOVResolver] DB refs — details={_LOV_DETAILS_REF!r}  master={_LOV_MASTER_REF!r}"
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STATIC FALLBACK INDEX  (lov_mapping_dynamic.json)
#
# Structure after inversion:
#   _STATIC_INV[TABLE][COLUMN][str(db_value)] = "Human Label"
#
# Keys normalised to UPPER so column-name case never causes a miss.
# Built once at import time; never mutated after that.
# ══════════════════════════════════════════════════════════════════════════════

_STATIC_INV: Dict[str, Dict[str, Dict[str, str]]] = {}


def _build_static_index(lov_data: dict) -> None:
    """
    Invert lov_mapping_dynamic.json:
        input  -> { table: { col: { "Label": db_value } } }
        output -> { TABLE: { COL: { str(db_value): "Label" } } }
    """
    global _STATIC_INV
    _STATIC_INV = {}
    for table, col_map in lov_data.items():
        t_key = table.upper()
        _STATIC_INV[t_key] = {}
        for col, label_to_val in col_map.items():
            c_key = col.upper()
            inv: Dict[str, str] = {}
            for label, db_val in label_to_val.items():
                inv[str(db_val)] = label
            _STATIC_INV[t_key][c_key] = inv

    total_cols = sum(len(v) for v in _STATIC_INV.values())
    log.info(
        f"[LOVResolver] Static index ready: "
        f"{len(_STATIC_INV)} tables, {total_cols} LOV columns."
    )


# Bootstrap at import time — resolve path relative to project root (one level
# above core/) so it works whether the server is launched from root or core/.
try:
    _base     = Path(__file__).resolve().parent.parent
    _lov_path = _base / "data" / "lov_mapping_dynamic.json"
    _raw: dict = json.loads(_lov_path.read_text(encoding="utf-8"))
    _build_static_index(_raw)
except Exception as _boot_err:
    log.warning(
        f"[LOVResolver] lov_mapping_dynamic.json not loaded ({_boot_err}). "
        "Will rely entirely on live DB lookups."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE DB LOOKUP  (COM_M_LOV + COM_L_LOV_DETAILS)
#
# Joins COM_L_LOV_DETAILS -> COM_M_LOV to get the canonical display name.
# One batched query for ALL unresolved values in the entire result-set.
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_from_db(conn, ids: Set[int]) -> Dict[str, str]:
    """
    Batch-fetch LOV labels for a set of numeric IDs.
    Returns { str(LOV_Detail_ID): "LOV_Detail_Name" }.
    Returns {} on any DB error — never raises.
    """
    if not ids:
        return {}

    try:
        placeholders = ", ".join(str(i) for i in ids)
        sql = (
            f"SELECT d.LOV_Details_ID, d.LOV_Details_Name "
            f"FROM  {_LOV_DETAILS_REF} d "
            f"INNER JOIN {_LOV_MASTER_REF} l ON l.LOV_ID = d.LOV_ID "
            f"WHERE d.LOV_Details_ID IN ({placeholders}) "
            f"  AND d.Status = 1"
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows   = cur.fetchall()
        result = {str(r[0]): r[1] for r in rows if r[1] is not None}
        log.info(
            f"[LOVResolver] DB lookup: {len(ids)} IDs queried, "
            f"{len(result)} labels returned."
        )
        return result

    except Exception as e:
        log.warning(f"[LOVResolver] DB lookup failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_int(val: Any):
    """Return int(val) if possible, else None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _static_label(table: str, col: str, int_val: int):
    """Look up a value in the static index. Returns label or None."""
    t_inv = _STATIC_INV.get(table.upper(), {})
    c_inv = t_inv.get(col.upper(), {})
    return c_inv.get(str(int_val))


def _distinct_numeric_pairs(result_rows: List[dict], col: str) -> List[Tuple[Any, int]]:
    """
    Return (raw_val, int_val) for every distinct numeric value in col.
    Non-numeric and None values are skipped.
    """
    seen: Set[int] = set()
    out = []
    for row in result_rows:
        raw = row.get(col)
        iv  = _to_int(raw)
        if iv is not None and iv not in seen:
            seen.add(iv)
            out.append((raw, iv))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def translate_lov_values(
    result_rows: List[dict],
    base_table:  str,
    conn,
) -> List[dict]:
    """
    Translate raw LOV IDs in result_rows to human-readable labels.

    Parameters
    ----------
    result_rows : list[dict]  -- raw rows from the query execution step
    base_table  : str         -- primary table the query ran against
    conn                      -- open DB connection

    Returns
    -------
    list[dict] with LOV IDs replaced by labels wherever a mapping exists.
    Columns / values with no mapping are returned exactly as received.
    The original list is never mutated.
    """
    if not result_rows:
        return result_rows

    try:
        all_cols = list(result_rows[0].keys())

        # col_map[col][str(int_val)] = "Label"  -- filled across both passes
        col_map: Dict[str, Dict[str, str]] = {}

        # IDs that pass 1 could not resolve -- sent to DB in one batch
        unresolved_ids: Set[int] = set()

        # ── PASS 1: static JSON index ─────────────────────────────────────────
        for col in all_cols:
            pairs = _distinct_numeric_pairs(result_rows, col)
            if not pairs:
                continue

            for raw_val, int_val in pairs:
                label = _static_label(base_table, col, int_val)
                if label:
                    col_map.setdefault(col, {})[str(int_val)] = label
                else:
                    unresolved_ids.add(int_val)

        # ── PASS 2: live DB for all unresolved IDs ────────────────────────────
        db_map: Dict[str, str] = {}
        if unresolved_ids and conn is not None:
            db_map = _fetch_from_db(conn, unresolved_ids)

        # Merge DB results back into col_map
        if db_map:
            for col in all_cols:
                pairs = _distinct_numeric_pairs(result_rows, col)
                for raw_val, int_val in pairs:
                    if str(int_val) in col_map.get(col, {}):
                        continue
                    label = db_map.get(str(int_val))
                    if label:
                        col_map.setdefault(col, {})[str(int_val)] = label

        # ── Diagnostics ───────────────────────────────────────────────────────
        resolved_cols   = [c for c, m in col_map.items() if m]
        unresolved_cols = [
            c for c in all_cols
            if _distinct_numeric_pairs(result_rows, c) and c not in col_map
        ]

        if resolved_cols:
            log.info(
                f"[LOVResolver] Resolved {len(resolved_cols)} LOV column(s): "
                f"{resolved_cols}"
            )
        if unresolved_cols:
            log.info(
                f"[LOVResolver] {len(unresolved_cols)} numeric column(s) had no LOV "
                f"mapping (kept as-is): {unresolved_cols}"
            )

        if not col_map:
            return result_rows

        # ── Apply translations row by row ─────────────────────────────────────
        translated: List[dict] = []
        for row in result_rows:
            new_row = dict(row)
            for col, val_to_label in col_map.items():
                raw = new_row.get(col)
                iv  = _to_int(raw)
                if iv is None:
                    continue
                label = val_to_label.get(str(iv))
                if label is not None:
                    new_row[col] = label
            translated.append(new_row)

        return translated

    except Exception as e:
        log.warning(
            f"[LOVResolver] Unexpected error, returning original rows: {e}",
            exc_info=True,
        )
        return result_rows
