"""
core/response_translator.py
─────────────────────────────────────────────────────────────────────────────
Post-execution translation layer.  Runs AFTER the DB returns rows.
Zero runtime DB queries — everything resolved from in-memory cache.

Translation order (must be respected):
  1. LOV   — status/enum integer codes  → human labels   (405 → "Released")
  2. FK    — foreign-key ID strings     → display names  (OP0001 → "Cutting")
  3. TYPE  — type-code strings          → type labels    (WORT0001 → "Production Work Order")
  4. HDR   — raw DB column names        → readable keys  (Work_Order_Status → "Work Order Status")

Design principles:
  • No DB calls at translation time — uses in-memory dict loaded at startup.
  • Falls back silently: if a value has no mapping it is left unchanged.
  • Header humanization is the LAST step so value translators can still
    find columns by their original DB names.
  • Cache refresh is handled externally (build_master_cache.py / scheduler).
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("NL2SQL")

# ── Paths (resolved relative to this file so it works regardless of cwd) ──────
_HERE        = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_CACHE_PATH  = _PROJECT_ROOT / "data" / "master_translation_cache.json"
_LOV_DYNAMIC_PATH = _PROJECT_ROOT / "data" / "lov_mapping_dynamic.json"
_CACHE_TTL_DAYS = 15

# ── In-memory stores ───────────────────────────────────────────────────────────
_lock = threading.RLock()

# Each store: COLUMN_NAME_UPPER → { raw_value_str → display_str }
# NOTE: _LOV_MAP is sourced directly from lov_mapping_dynamic.json via
# load_lov_map() — NOT from master_translation_cache.json. This keeps LOV
# translation independent of build_master_cache.py, so a fresh extractor run
# reflects in output translation immediately after a reload, no cache
# rebuild round-trip required. _FK_MAP / _TYPE_MAP still come from the
# master cache, since those require a DB join-graph walk to resolve.
_LOV_MAP:  Dict[str, Dict[str, str]] = {}
_FK_MAP:   Dict[str, Dict[str, str]] = {}
_TYPE_MAP: Dict[str, Dict[str, str]] = {}

_cache_loaded_at:  Optional[datetime] = None
_cache_expires_at: Optional[datetime] = None
_lov_loaded_at:    Optional[datetime] = None


# ══════════════════════════════════════════════════════════════════════════════
# Public API — startup
# ══════════════════════════════════════════════════════════════════════════════

def load_translation_cache(cache_path: Optional[Path] = None) -> bool:
    """
    Load master_translation_cache.json into memory.
    Called once during backend startup (lifespan).
    Returns True on success, False if file missing/corrupt (non-fatal).
    """
    global _cache_loaded_at, _cache_expires_at

    path = Path(cache_path) if cache_path else _CACHE_PATH

    if not path.exists():
        log.warning(
            "[translator] master_translation_cache.json not found at %s — "
            "run build_master_cache.py to generate it. "
            "FK/Type translation will be skipped until cache is built.",
            path,
        )
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[translator] Failed to read translation cache: %s", exc)
        return False

    with _lock:
        _FK_MAP.clear()
        _TYPE_MAP.clear()

        _FK_MAP.update(data.get("fk", {}))
        _TYPE_MAP.update(data.get("type", {}))

        generated = data.get("generated_at", "unknown")
        expires   = data.get("expires_at",   "unknown")

        try:
            _cache_loaded_at  = datetime.fromisoformat(generated)
            _cache_expires_at = datetime.fromisoformat(expires)
        except Exception:
            _cache_loaded_at  = datetime.utcnow()
            _cache_expires_at = None

        stats = data.get("stats", {})

    log.info(
        "[translator] Translation cache loaded — "
        "fk=%d  type=%d columns | generated=%s expires=%s",
        len(_FK_MAP), len(_TYPE_MAP), generated, expires,
    )

    # Warn if cache is already stale
    if _cache_expires_at and datetime.utcnow() > _cache_expires_at:
        log.warning(
            "[translator] ⚠️  Translation cache EXPIRED (was due %s). "
            "Run build_master_cache.py to refresh.",
            expires,
        )

    return True


def _flatten_lov_data(lov_data: dict) -> Dict[str, Dict[str, str]]:
    """
    Flatten lov_mapping_dynamic.json's
        { table: { column: { raw_value: {"description": label} } } }
    shape into the flat shape _apply_lov expects:
        { COLUMN_NAME_UPPER: { raw_value_str: label } }
    Mirrors the merge logic that used to live in build_master_cache.py's
    lov_cache builder, so behavior (including the "first table wins per
    column name" merge across tables) stays identical.
    """
    flat: Dict[str, Dict[str, str]] = {}
    for _tbl, col_map in lov_data.items():
        if not isinstance(col_map, dict):
            continue
        for col, mapping in col_map.items():
            if not isinstance(mapping, dict):
                continue
            col_upper = col.upper()
            bucket = flat.setdefault(col_upper, {})
            for raw_value, meta in mapping.items():
                if isinstance(meta, dict):
                    label = meta.get("description") or str(raw_value)
                else:
                    label = str(meta)  # tolerate a flat {value: label} shape too
                bucket[str(raw_value)] = label
    return flat


def load_lov_map(path: Optional[Path] = None) -> bool:
    """
    Load LOV translations directly from lov_mapping_dynamic.json — the
    extractor's raw output — independent of master_translation_cache.json.

    This means output-side LOV translation reflects whatever lov_extractor.py
    last wrote to disk, with no build_master_cache.py round-trip needed.
    Called once at startup and again by reload_lov_data()'s admin hook, so a
    single /admin/reload-lov call refreshes both the input-side LOV_DATA and
    this output-side map from the same file.

    Returns True on success, False if file missing/corrupt (non-fatal —
    LOV translation is simply skipped until the file is available).
    """
    global _lov_loaded_at

    lov_path = Path(path) if path else _LOV_DYNAMIC_PATH

    if not lov_path.exists():
        log.warning(
            "[translator] lov_mapping_dynamic.json not found at %s — "
            "LOV translation will be skipped.",
            lov_path,
        )
        return False

    try:
        lov_data = json.loads(lov_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[translator] Failed to read lov_mapping_dynamic.json: %s", exc)
        return False

    flat = _flatten_lov_data(lov_data)

    with _lock:
        _LOV_MAP.clear()
        _LOV_MAP.update(flat)
        _lov_loaded_at = datetime.utcnow()

    log.info("[translator] LOV map loaded from lov_mapping_dynamic.json — %d column(s).", len(_LOV_MAP))
    return True


def cache_status() -> dict:
    """Return a status dict suitable for /health endpoint."""
    with _lock:
        return {
            "loaded":        bool(_LOV_MAP or _FK_MAP or _TYPE_MAP),
            "lov_loaded_at": _lov_loaded_at.isoformat() if _lov_loaded_at else None,
            "generated_at":  _cache_loaded_at.isoformat() if _cache_loaded_at else None,
            "expires_at":    _cache_expires_at.isoformat() if _cache_expires_at else None,
            "stale":         bool(_cache_expires_at and datetime.utcnow() > _cache_expires_at),
            "lov_columns":   len(_LOV_MAP),
            "fk_columns":    len(_FK_MAP),
            "type_columns":  len(_TYPE_MAP),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Translation steps (internal)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_lov(rows: List[dict]) -> List[dict]:
    """
    Step 1 — LOV translation.
    Replaces integer status/enum values with their human labels.
    Example: Work_Order_Status=405 → Work_Order_Status="Released"
    """
    if not _LOV_MAP or not rows:
        return rows

    with _lock:
        lov_snapshot = dict(_LOV_MAP)

    translated = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            col_upper = col.upper()
            if col_upper in lov_snapshot and val is not None:
                mapped = lov_snapshot[col_upper].get(str(val))
                new_row[col] = mapped if mapped is not None else val
            else:
                new_row[col] = val
        translated.append(new_row)

    return translated


def _apply_fk(rows: List[dict]) -> List[dict]:
    """
    Step 2 — FK translation.
    Replaces foreign-key ID strings with display names from master tables.
    Example: Operation_ID="OP0001" → Operation_ID="Cutting"
    Only runs on columns whose name ends with _ID (case-insensitive).
    """
    if not _FK_MAP or not rows:
        return rows

    with _lock:
        fk_snapshot = dict(_FK_MAP)

    translated = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            col_upper = col.upper()
            # Only attempt FK translation on _ID columns that have a mapping
            if col_upper.endswith("_ID") and col_upper in fk_snapshot and val is not None:
                mapped = fk_snapshot[col_upper].get(str(val).strip())
                new_row[col] = mapped if mapped is not None else val
            else:
                new_row[col] = val
        translated.append(new_row)

    return translated


def _apply_type(rows: List[dict]) -> List[dict]:
    """
    Step 3 — Type code translation.
    Replaces type-code values (e.g. WORT0001) with type labels.
    Example: Work_Order_Type_ID="WORT0001" → Work_Order_Type_ID="Production Work Order"
    Uses the separate TYPE map so type tables don't collide with FK tables.
    """
    if not _TYPE_MAP or not rows:
        return rows

    with _lock:
        type_snapshot = dict(_TYPE_MAP)

    translated = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            col_upper = col.upper()
            if col_upper in type_snapshot and val is not None:
                mapped = type_snapshot[col_upper].get(str(val).strip())
                new_row[col] = mapped if mapped is not None else val
            else:
                new_row[col] = val
        translated.append(new_row)

    return translated


# ── Header suffix overrides ───────────────────────────────────────────────────
# These exact column suffixes get custom labels instead of the auto-generated
# title-case version.  Add domain-specific overrides here as needed.
_HEADER_OVERRIDES: Dict[str, str] = {
    "ID":         "ID",          # keep "ID" uppercase
    "SN":         "Serial No",
    "SN_ROW_ID":  "Row ID",
    "OP_ID":      "Operation ID",
    "WO_ID":      "Work Order ID",
    "BOM_ID":     "BOM ID",
    "MFG":        "Manufacturing",
    "QTY":        "Quantity",
    "UOM":        "Unit of Measure",
    "ERP":        "ERP",
    "PO":         "Purchase Order",
    "SO":         "Sales Order",
}

# Table-name prefixes to strip before humanizing column headers.
# Driven by TABLE_NAME_PREFIXES or TABLE_PREFIX env vars so adding a new
# module (e.g. ALM) requires only an env change, not a code change.
def _build_table_prefixes_to_strip() -> tuple:
    """
    Build the tuple of prefixes to strip from Table.Column headers.

    Sources (first non-empty wins):
      1. TABLE_NAME_PREFIXES  e.g. "MFC_M_,MFC_L_,MFC_T_,ALM_M_,ALM_L_"
      2. TABLE_PREFIX         e.g. "MFC,ALM,USR"
         → expanded to "<CODE>_M_", "<CODE>_L_", "<CODE>_T_", "<CODE>_"
      3. Hard fallback        → ("MFC_M_", "MFC_L_", "MFC_T_", "MFC_")
    """
    raw_name = os.getenv("TABLE_NAME_PREFIXES", "").strip()
    if raw_name:
        prefixes = tuple(p.strip() for p in raw_name.split(",") if p.strip())
        if prefixes:
            return prefixes

    raw_code = os.getenv("TABLE_PREFIX", "").strip()
    if raw_code:
        codes = [c.strip().upper() for c in raw_code.split(",") if c.strip()]
        return tuple(
            f"{c}{sfx}"
            for c in codes
            for sfx in ("_M_", "_L_", "_T_", "_")
        )

    return ("MFC_M_", "MFC_L_", "MFC_T_", "MFC_")

_TABLE_PREFIXES_TO_STRIP: tuple = _build_table_prefixes_to_strip()


def _humanize_column(col: str) -> str:
    """
    Convert a raw DB column name to a readable display header.

    Rules (in order):
      1. Strip table-name prefix if col was returned as Table.Column
      2. Check _HEADER_OVERRIDES for exact match on the raw upper value
      3. Split on underscore, title-case each token
      4. Re-join with spaces
      5. Apply token-level overrides (ID stays ID, QTY→Quantity etc.)

    Examples:
      Work_Order_Status         → "Work Order Status"
      MFC_M_WORK_ORDER.WO_ID    → "WO ID"  (then override → "Work Order ID")
      Work_Order_Sn_Row_ID      → "Work Order Row ID"
      Operation_ID              → "Operation ID"
    """
    # Strip table prefix — two cases:
    #   1. "Table.Column" format → take the part after the dot
    #   2. Column name that starts with a known module prefix (no dot)
    #      e.g. "MFC_M_WORK_ORDER_STATUS" → strip "MFC_M_" → "WORK_ORDER_STATUS"
    if "." in col:
        col = col.split(".")[-1]
    else:
        for pfx in sorted(_TABLE_PREFIXES_TO_STRIP, key=len, reverse=True):
            if col.upper().startswith(pfx.upper()):
                col = col[len(pfx):]
                break

    # Exact override on raw upper
    if col.upper() in _HEADER_OVERRIDES:
        return _HEADER_OVERRIDES[col.upper()]

    # Split tokens and apply token-level overrides
    tokens = re.split(r"[_\s]+", col)
    human_tokens = []
    for tok in tokens:
        if not tok:
            continue
        upper_tok = tok.upper()
        if upper_tok in _HEADER_OVERRIDES:
            human_tokens.append(_HEADER_OVERRIDES[upper_tok])
        else:
            human_tokens.append(tok.capitalize())

    return " ".join(human_tokens)


def _apply_headers(rows: List[dict]) -> List[dict]:
    """
    Step 4 — Header humanization.
    Renames all dict keys in every row to human-readable display names.
    This MUST run last — previous steps rely on original column names.

    Example:
      {"Work_Order_Status": "Released", "Operation_ID": "Cutting"}
      → {"Work Order Status": "Released", "Operation": "Cutting"}
    """
    if not rows:
        return rows

    # Build the rename map once from the first row's keys
    first_row_keys = list(rows[0].keys())
    rename_map: Dict[str, str] = {}
    seen_display: Dict[str, int] = {}   # handle duplicate display names

    for raw_key in first_row_keys:
        display = _humanize_column(raw_key)
        # Deduplicate: if two raw cols map to same display, append suffix
        if display in seen_display:
            seen_display[display] += 1
            display = f"{display} {seen_display[display]}"
        else:
            seen_display[display] = 1
        rename_map[raw_key] = display

    return [{rename_map[k]: v for k, v in row.items()} for row in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Public API — main translation entry point
# ══════════════════════════════════════════════════════════════════════════════

def translate_rows(
    rows: List[dict],
    *,
    apply_lov:     bool = True,
    apply_fk:      bool = True,
    apply_type:    bool = True,
    apply_headers: bool = True,
) -> List[dict]:
    """
    Run all four translation passes on a list of row dicts.

    Order: LOV → FK → TYPE → HEADERS

    Parameters
    ----------
    rows          : raw rows from DB execution
    apply_lov     : toggle LOV (status/enum) translation
    apply_fk      : toggle FK ID → name translation
    apply_type    : toggle type-code translation
    apply_headers : toggle header humanization

    Returns the translated list (original list is NOT mutated).
    """
    if not rows:
        return rows

    if apply_lov:
        rows = _apply_lov(rows)

    if apply_fk:
        rows = _apply_fk(rows)

    if apply_type:
        rows = _apply_type(rows)

    if apply_headers:
        rows = _apply_headers(rows)

    return rows