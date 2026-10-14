from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("meta_loader")

_HERE    = Path(__file__).resolve().parent   # core/
_PROJECT = _HERE.parent                      # project root
_DATA    = _PROJECT / "data"                 # data/module_tables.json, data/*_narrow.json

# ── Module discovery ──────────────────────────────────────────────────────────

_MODULE_TABLES_FILE = "module_tables.json"

# Cached result so we don't re-read the file on every call
_MODULE_TABLES_CACHE: List[dict] | None = None


def _env_module_prefixes() -> List[str]:
    """Return active module codes from TABLE_PREFIX."""
    raw = os.getenv("TABLE_PREFIX", "")
    return [m.strip().upper() for m in raw.split(",") if m.strip()]


def _load_module_tables(data_dir: Path = _DATA) -> List[dict]:
    """
    Read data/module_tables.json once and cache the result.
    Returns the list of module entries exactly as written in the JSON.
    """
    global _MODULE_TABLES_CACHE
    if _MODULE_TABLES_CACHE is not None:
        return _MODULE_TABLES_CACHE

    path = data_dir / _MODULE_TABLES_FILE
    if not path.exists():
        log.warning(
            "[meta_loader] %s not found — no modules will be loaded. "
            "Create data/%s to register modules.",
            path, _MODULE_TABLES_FILE,
        )
        _MODULE_TABLES_CACHE = []
        return _MODULE_TABLES_CACHE

    try:
        _MODULE_TABLES_CACHE = json.loads(path.read_text(encoding="utf-8"))
        log.info(
            "[meta_loader] module_tables.json loaded — %d module(s): %s",
            len(_MODULE_TABLES_CACHE),
            [m.get("id") for m in _MODULE_TABLES_CACHE],
        )
    except Exception as exc:
        log.error("[meta_loader] Failed to load %s: %s", path, exc)
        _MODULE_TABLES_CACHE = []

    return _MODULE_TABLES_CACHE


def _get_module_prefixes(data_dir: Path = _DATA) -> List[str]:
    """
    Return the ordered list of active module codes.

    TABLE_PREFIX in .env is the single source of truth — no module_tables.json
    lookup, no hardcoded fallback module. `data_dir` is accepted only for call-
    site compatibility with load_all_meta()/load_module_meta(), which still use
    module_tables.json to resolve narrow-file paths for whatever modules
    TABLE_PREFIX names.
    """
    modules = _env_module_prefixes()
    if not modules:
        log.warning(
            "[meta_loader] TABLE_PREFIX is empty/unset in .env — no active modules."
        )
    return modules


def _get_table_name_prefixes() -> List[str]:
    """
    Read TABLE_NAME_PREFIXES from .env / environment.
    Returns a list of uppercase prefix strings.
    e.g.  TABLE_NAME_PREFIXES=MFC_M_,MFC_L_,MFC_T_,ACM_,QMS_,INV_
           →  ['MFC_M_', 'MFC_L_', 'MFC_T_', 'ACM_', 'QMS_', 'INV_']
    """
    active_modules = _get_module_prefixes()
    active_heads = tuple(f"{m.upper()}_" for m in active_modules)
    raw = os.getenv("TABLE_NAME_PREFIXES", "")
    prefixes = [
        p.strip().upper()
        for p in raw.split(",")
        if p.strip() and (not active_heads or p.strip().upper().startswith(active_heads))
    ]
    seen = set(prefixes)
    for module in active_modules:
        for tier in ("M", "L", "T"):
            pfx = f"{module}_{tier}_"
            if pfx not in seen:
                prefixes.append(pfx)
                seen.add(pfx)
    return prefixes


def _resolve_narrow_path(module_entry: dict, data_dir: Path) -> Path:
    """
    Determine the _narrow.json path for a module entry.

    Priority:
      1. Explicit "narrow_file" field in the entry  (e.g. "/data/mfc_narrow.json")
      2. Default convention: data/<id.lower()>_narrow.json
    """
    if "narrow_file" in module_entry:
        # Treat absolute-looking paths as relative to data_dir root
        nf = module_entry["narrow_file"].lstrip("/")
        # Strip leading "data/" if present, since data_dir already points there
        if nf.startswith("data/"):
            nf = nf[len("data/"):]
        return data_dir / nf

    module_id = module_entry.get("id", "").lower()
    return data_dir / f"{module_id}_narrow.json"


def load_all_meta(data_dir: Path = _DATA) -> Dict[str, dict]:
    """
    Load and merge all *_narrow.json files discovered via data/module_tables.json.

    Each narrow file has the shape:
        {"modules": {"MFC": {"subgroups": {...}}}}

    Returns a single merged dict with ALL modules under the "modules" key:
        {"modules": {"MFC": {...}, "USR": {...}, "ALM": {...}}}

    Uses a deep merge on the "modules" key so that loading USR does NOT
    overwrite MFC (which a plain dict.update() would do).

    Skips missing files with a warning so one absent file doesn't break startup.
    """
    modules_raw = _load_module_tables(data_dir)
    active_modules = set(_get_module_prefixes(data_dir))

    # Build (module_id, narrow_path) pairs from module_tables.json
    module_entries: List[tuple[str, Path]] = []
    for entry in modules_raw:
        mid = (entry.get("id") or "").strip().upper()
        if not mid:
            continue
        if active_modules and mid not in active_modules:
            continue
        module_entries.append((mid, _resolve_narrow_path(entry, data_dir)))

    if not module_entries:
        # Fallback: TABLE_PREFIX env var
        for mid in _get_module_prefixes(data_dir):
            module_entries.append((mid, data_dir / f"{mid.lower()}_narrow.json"))

    merged: Dict[str, dict] = {"modules": {}}

    for module, fname in module_entries:
        if not fname.exists():
            log.warning(
                "[meta_loader] %s not found — skipping module %s.",
                fname, module,
            )
            continue
        try:
            data = json.loads(fname.read_text(encoding="utf-8"))

            incoming_modules = data.get("modules", {})

            # Inject runtime metadata so downstream consumers never need to
            # hardcode module IDs:
            #   module_prefix  →  TABLE_PREFIX code (e.g. "USR")
            #   narrow_source  →  filename stem     (e.g. "usr_narrow")
            narrow_source = fname.stem
            for mod_val in incoming_modules.values():
                if isinstance(mod_val, dict):
                    mod_val.setdefault("module_prefix", module)
                    mod_val.setdefault("narrow_source", narrow_source)

            merged["modules"].update(incoming_modules)

            # Preserve any other top-level keys (e.g. "version", "metadata")
            for k, v in data.items():
                if k != "modules":
                    merged.setdefault(k, v)

            table_count = sum(
                len(sg.get("tables", sg) if isinstance(sg, dict) else sg)
                for mod in incoming_modules.values()
                for sg in mod.get("subgroups", {}).values()
            )
            log.info(
                "[meta_loader] Loaded %s → %d tables (modules=%s prefix=%s source=%s)",
                fname.name, table_count, list(incoming_modules.keys()), module, narrow_source,
            )
        except Exception as exc:
            log.error("[meta_loader] Failed to load %s: %s", fname, exc)

    total_tables = sum(
        len(sg.get("tables", sg) if isinstance(sg, dict) else sg)
        for mod in merged["modules"].values()
        for sg in mod.get("subgroups", {}).values()
    )
    log.info(
        "[meta_loader] Total META tables loaded: %d across %d module(s): %s",
        total_tables, len(merged["modules"]), ", ".join(merged["modules"].keys()),
    )
    return merged


def load_module_meta(module: str, data_dir: Path = _DATA) -> Dict[str, dict]:
    """
    Load meta for a single module by name.
    Checks module_tables.json for an explicit narrow_file override first.
    """
    # Check if module_tables.json declares a custom narrow_file for this module
    for entry in _load_module_tables(data_dir):
        if (entry.get("id") or "").upper() == module.upper():
            fname = _resolve_narrow_path(entry, data_dir)
            break
    else:
        fname = data_dir / f"{module.lower()}_narrow.json"

    if not fname.exists():
        log.warning("[meta_loader] %s not found.", fname)
        return {}
    try:
        data = json.loads(fname.read_text(encoding="utf-8"))
        table_count = sum(
            len(sg.get("tables", sg) if isinstance(sg, dict) else sg)
            for mod in data.get("modules", {}).values()
            for sg in mod.get("subgroups", {}).values()
        )
        log.info("[meta_loader] Loaded %s → %d tables", fname.name, table_count)
        return data
    except Exception as exc:
        log.error("[meta_loader] Failed to load %s: %s", fname, exc)
        return {}


def get_module_prefixes() -> List[str]:
    """Public accessor — returns module IDs from module_tables.json."""
    return _get_module_prefixes()


def get_table_name_prefixes() -> List[str]:
    """Public accessor for TABLE_NAME_PREFIXES env var."""
    return _get_table_name_prefixes()


def get_stop_parts() -> set[str]:
    """
    Build the _STOP_PARTS set for literal_resolver dynamically.
    Combines:
      - generic structural tokens (always noise in table names)
      - module prefix tokens derived from module_tables.json (or TABLE_PREFIX fallback)
      - sub-prefix tokens derived from TABLE_NAME_PREFIXES env var
    """
    generic = {
        "t", "m", "l", "id", "op", "sn", "rou", "his",
        "history", "log", "audit", "temp", "tmp",
    }

    module_tokens = {mod.lower() for mod in _get_module_prefixes()}

    prefix_tokens: set[str] = set()
    for pfx in _get_table_name_prefixes():
        for part in pfx.split("_"):
            tok = part.strip().lower()
            if tok:
                prefix_tokens.add(tok)

    stop_parts = generic | module_tokens | prefix_tokens
    log.debug("[meta_loader] stop_parts=%s", stop_parts)
    return stop_parts