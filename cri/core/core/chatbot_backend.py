import ast
import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from collections import deque
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import pytz
import redis
import sqlglot
import sqlglot.errors as _sqlglot_errors
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response as _FResponse, StreamingResponse as _FStreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ── Prometheus metrics ──
_METRICS_ENABLED: bool = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _METRICS_ENABLED = True
except ImportError:
    pass

if _METRICS_ENABLED:
    _METRIC_REQUESTS_TOTAL = Counter(
        "nl2sql_requests_total",
        "Total number of /query requests",
        ["intent", "cache_hit", "error"],
    )
    _METRIC_REQUEST_LATENCY = Histogram(
        "nl2sql_request_latency_seconds",
        "End-to-end /query latency (seconds)",
        ["intent"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _METRIC_LLM_LATENCY = Histogram(
        "nl2sql_llm_latency_seconds",
        "Latency of LLM HTTP call (seconds)",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    )
    _METRIC_LLM_ERRORS = Counter(
        "nl2sql_llm_errors_total",
        "Total LLM call failures",
        ["reason"],
    )
    _METRIC_LLM_CIRCUIT_STATE = Gauge(
        "nl2sql_llm_circuit_breaker_open",
        "1 if LLM circuit breaker is OPEN or HALF_OPEN, 0 if CLOSED",
    )
    _METRIC_DB_LATENCY = Histogram(
        "nl2sql_db_latency_seconds",
        "Latency of SQL execution (seconds)",
        ["outcome"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )
    _METRIC_DB_ERRORS = Counter(
        "nl2sql_db_errors_total",
        "Total DB execution failures",
    )
    _METRIC_CACHE_HITS = Counter(
        "nl2sql_cache_hits_total",
        "Cache hit count",
        ["kind"],
    )
    _METRIC_CACHE_MISSES = Counter(
        "nl2sql_cache_misses_total",
        "Cache miss count",
    )

# -------------------- Rate Limiting & Queue --------------------
_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
_request_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
_SEMAPHORE_TIMEOUT = float(os.getenv("SEMAPHORE_TIMEOUT_SECS", "30.0"))

_RATE_LIMIT_STR: str = os.getenv("RATE_LIMIT", "10/minute")
_TRUSTED_PROXIES: set = {
    p.strip() for p in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if p.strip()
}

def _parse_rate(rate_str: str) -> tuple:
    count, period = rate_str.strip().split("/")
    seconds = {"second": 1, "minute": 60, "hour": 3600}.get(period.strip(), 60)
    return int(count), seconds

_RATE_LIMIT_COUNT, _RATE_LIMIT_WINDOW = _parse_rate(_RATE_LIMIT_STR)

_RATE_LIMIT_LUA = """
local key = KEYS[1]
local wstart = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local member = ARGV[3]
local limit = tonumber(ARGV[4])
local window = tonumber(ARGV[5])
redis.call('ZREMRANGEBYSCORE', key, '-inf', wstart)
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, window + 1)
local count = redis.call('ZCARD', key)
return count
"""
_rl_sha: Optional[str] = None

def _get_real_ip(request: "Request") -> str:
    peer = request.client.host if request.client else "unknown"
    if peer in _TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            ips = [x.strip() for x in forwarded.split(",")]
            for ip in reversed(ips):
                if ip not in _TRUSTED_PROXIES:
                    return ip
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    return peer

def _load_rl_script(redis_client) -> Optional[str]:
    global _rl_sha
    try:
        _rl_sha = redis_client.script_load(_RATE_LIMIT_LUA)
        log.info(f"[rate_limit] Lua script loaded, sha={_rl_sha[:12]}…")
        return _rl_sha
    except Exception as exc:
        log.warning(f"[rate_limit] Failed to load Lua script ({exc})")
        return None

def _check_rate_limit_redis(ip: str, redis_client) -> bool:
    global _rl_sha
    try:
        now_ms = int(time.time() * 1000)
        window_ms = _RATE_LIMIT_WINDOW * 1000
        wstart_ms = now_ms - window_ms
        member = f"{now_ms}:{secrets.token_hex(4)}"
        key = f"nl2sql:rl:{ip}"

        def _run(sha):
            return redis_client.evalsha(
                sha, 1, key,
                wstart_ms, now_ms, member,
                _RATE_LIMIT_COUNT, _RATE_LIMIT_WINDOW + 1,
            )

        if _rl_sha is None:
            _load_rl_script(redis_client)

        try:
            count = _run(_rl_sha)
        except redis.exceptions.NoScriptError:
            log.warning("[rate_limit] Lua SHA stale, reloading…")
            _load_rl_script(redis_client)
            count = _run(_rl_sha)

        return int(count) <= _RATE_LIMIT_COUNT

    except Exception as exc:
        log.warning(f"[rate_limit] Redis check failed ({exc}) — fail-open")
        return True

def _check_rate_limit(ip: str) -> bool:
    if pipeline is not None and pipeline._cache._enabled:
        return _check_rate_limit_redis(ip, pipeline._cache._redis)
    return True

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# ============================================================
# LOGGING — JSON to file, clean plain text to terminal
# ============================================================
_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
_LOG_FILE: str = os.getenv(
    "LOG_FILE",
    str(Path(__file__).resolve().parent.parent / "logs" / "nl2sql.log"),
)

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        skip = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "exc_info", "exc_text", "stack_info",
        }
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)
    
def _configure_logging() -> logging.Logger:
    level = getattr(logging, _LOG_LEVEL, logging.INFO)

    log_path = Path(_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    return logging.getLogger("NL2SQL")

log = _configure_logging()

# ── Terminal handler:───────
from .terminal_logger import attach_terminal_handler
attach_terminal_handler()

# ── VS Terminal coloured status printer ──────────────────────────────────────
_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "grey": "\033[90m",
    "magenta": "\033[95m",
}

_VERBOSE_TERMINAL: bool = os.getenv("VERBOSE_TERMINAL", "true").lower() in ("1", "true", "yes")

def _vprint(label: str, colour: str, msg: str, detail: str = "") -> None:
    log.info(
        f"[{label}] {msg}" + (f" {detail}" if detail else ""),
        extra={"vprint": True, "vlabel": label, "vdetail": detail},
    )
    if not _VERBOSE_TERMINAL:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    badge = f"{_C[colour]}{_C['bold']}[{label}]{_C['reset']}"
    det = f" {_C['grey']}{detail}{_C['reset']}" if detail else ""
    print(f"{_C['grey']}{ts}{_C['reset']} {badge} {msg}{det}", flush=True)

# ============================================================
# CONFIG
# ============================================================
_LLM_BACKEND: str = os.getenv(
    "LLM_BACKEND",
    "llamacpp"
).strip().lower()

MODEL_PATH: str = os.getenv(
    "LLM_MODEL_PATH",
    ""
).strip()
if _LLM_BACKEND == "llamacpp":
    if not MODEL_PATH:
        raise RuntimeError(
            "LLM_MODEL_PATH must be set when LLM_BACKEND=llamacpp"
        )

elif _LLM_BACKEND == "ollama":
    pass

else:
    raise RuntimeError(
        f"Unsupported LLM_BACKEND={_LLM_BACKEND!r}. "
        "Expected 'ollama' or 'llamacpp'."
    )

_DATA = _PROJECT_ROOT / "data"
_REQUIRED_FILES = [
    "lov_mapping_dynamic.json",
]
_missing = [f for f in _REQUIRED_FILES if not (_DATA / f).exists()]
if _missing:
    raise FileNotFoundError(f"Missing required data files: {', '.join(_missing)}")

# ── Meta loading — driven by TABLE_PREFIX in .env ────────────────────────────
from core.meta_loader import (
    load_all_meta as _load_all_meta,
    get_module_prefixes as _get_module_prefixes,
    get_table_name_prefixes as _get_table_name_prefixes_loader,
)

# MODULE_DATA: merged narrow JSON dict used by col-role index, scope guard, table selector
MODULE_DATA: dict = _load_all_meta(_DATA)
_MODULE_PROMPT_RULES_DIR = _DATA / "module_prompt_rules"
_MODULE_PROMPT_RULES_LEGACY_PATH = _DATA / "module_prompt_rules.json"

def _parse_rule_value(v) -> str:
    if isinstance(v, list):
        return "\n".join(str(line) for line in v)
    return str(v)

_GLOBAL_RULE_KEY_PREFIX = "_global"

_MODULE_TO_FILE_KEY: Dict[str, str] = {}

_GLOBAL_RULES_BY_FILE_KEY: Dict[str, str] = {}

def _load_module_prompt_rules_from_dir(dir_path: Path) -> Dict[str, str]:
    """
    Load one rule-text entry per REAL module (items, bom, routing,
    erp_operation, la_operation, uom, mbom_operations, ...) from
    data/module_prompt_rules/*.json, AND one combined rule-text entry per
    CONTAINER file (e.g. "prod"), for module-only lookups.

    Each file is a CONTAINER for a group of modules — e.g.
    mfc_prompt_rules.json holds the rules for every module under the mfc
    schema, keyed by module name at the JSON top level:
        { "items": [...], "bom": [...], "routing": [...], ... }
    prd_prompt_rules.json would hold the equivalent set for the prd schema,
    qal_prompt_rules.json for qal, and so on.

    # MODULE_PROMPT_RULES supports both:
    #
    # 1. Real submodule keys:
    #       items, bom, routing, work_order, ...
    #
    # 2. Container/module keys:
    #       prod, mfc, ...
    #
    # For module-only flows where subgroup_name is empty,
    # the container-level key is used.

    The file's inner keys are flattened directly into MODULE_PROMPT_RULES,
    because that's what's actually looked up at query time
    (MODULE_PROMPT_RULES.get(module_name, "")) — module_name is the real
    module the question was routed to (e.g. "items"), NOT the container
    file's name (e.g. "mfc"). So "items" pulls only the "items" block out
    of whichever container file defines it — never the whole file, never
    another module's block. The container file's own stem (e.g. "prod")
    is also registered as a key, holding all of that file's module blocks
    concatenated together — this is what a module-only lookup (no
    subgroup) resolves to.

    Keys starting with "_global" (e.g. "_global_hierarchy") are special:
    they are NOT real modules and are never returned on their own. They
    hold cross-module bridging guidance for every module in that same
    container file (e.g. how the old MFC/PRD/SEQ layers interconnect) and
    are recorded in _GLOBAL_RULES_BY_FILE_KEY, then automatically prepended
    to every real module's rule text from that same file — see
    _get_module_prompt_rules. This is what lets an "items" question still
    carry the MFC<->PRD<->SEQ bridging rules, a "pickslip" question too,
    etc., without duplicating that text into every module block by hand.

    If two container files define the same module key (e.g. both mfc and
    prd happen to have an "items" entry), the later file (alphabetical)
    wins and a warning is logged — module keys are expected to be unique
    across files. Rename one of them if this fires unexpectedly.

    Back-compat: if a file's JSON root is a flat list/string instead of an
    object keyed by module, it's loaded as a single rule block under the
    file's own stem (so a file with no internal module keys still works
    exactly like before).
    """
    _SUFFIX = "_prompt_rules"
    rules: Dict[str, str] = {}
    _MODULE_TO_FILE_KEY.clear()
    _GLOBAL_RULES_BY_FILE_KEY.clear()
    for path in sorted(dir_path.glob("*.json")):
        stem = path.stem
        file_key = stem[: -len(_SUFFIX)] if stem.lower().endswith(_SUFFIX) else stem
        file_key = file_key.lower()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                container_parts: List[str] = []

                for k, v in raw.items():
                    module_key = str(k).strip().lower()
                    rule_text = _parse_rule_value(v)

                    if module_key.startswith(_GLOBAL_RULE_KEY_PREFIX):
                        _GLOBAL_RULES_BY_FILE_KEY[file_key] = rule_text
                        continue

                    # Keep the individual rule block available.
                    if module_key in rules:
                        log.warning(
                            "[startup] Module key %r from %s overwrites a rule already "
                            "loaded from another module_prompt_rules file — module keys "
                            "must be unique across files",
                            module_key, path,
                        )

                    rules[module_key] = rule_text
                    _MODULE_TO_FILE_KEY[module_key] = file_key

                    # Also collect it for module-level lookup.
                    container_parts.append(
                        f"\n=== {module_key.upper()} RULES ===\n{rule_text}"
                    )

            
                if container_parts:
                    rules[file_key] = "\n".join(container_parts)
                    _MODULE_TO_FILE_KEY[file_key] = file_key
            else:
                rules[file_key] = _parse_rule_value(raw)
                _MODULE_TO_FILE_KEY[file_key] = file_key
        except Exception as exc:
            log.error("[startup] Failed to load %s: %s — skipping", path, exc)
    if _GLOBAL_RULES_BY_FILE_KEY:
        log.info(
            "[startup] Loaded cross-module hierarchy/bridging rules for container(s): %s "
            "— auto-prepended to every module sourced from that same file",
            list(_GLOBAL_RULES_BY_FILE_KEY.keys()),
        )
    return rules

def _load_module_prompt_rules(dir_path: Path, legacy_path: Path) -> Dict[str, str]:
    if dir_path.is_dir():
        rules = _load_module_prompt_rules_from_dir(dir_path)
        if rules:
            log.info(
                "[startup] Loaded module prompt rules for %d module(s) from %s: %s",
                len(rules), dir_path, list(rules.keys()),
            )
            return rules
        log.warning(
            "[startup] %s exists but has no readable *.json files — "
            "module-scoped prompt rules disabled",
            dir_path,
        )
        return {}

    # Fallback: legacy single-file format, kept only for migration convenience.
    if legacy_path.exists():
        log.warning(
            "[startup] %s not found — falling back to legacy single-file %s. "
            "Migrate to the per-module directory when convenient.",
            dir_path, legacy_path,
        )
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("module_prompt_rules.json must be a flat object of module_key -> rule text")
            rules = {str(k).lower(): _parse_rule_value(v) for k, v in raw.items()}
            log.info(
                "[startup] Loaded module prompt rules for %d module(s): %s",
                len(rules), list(rules.keys()),
            )
            return rules
        except Exception as exc:
            log.error("[startup] Failed to load %s: %s — module-scoped rules disabled", legacy_path, exc)
            return {}

    log.warning(
        "[startup] Neither %s nor %s found — "
        "module-scoped prompt rules disabled (LLM will get no domain hierarchy guidance)",
        dir_path, legacy_path,
    )
    return {}


MODULE_PROMPT_RULES: Dict[str, str] = _load_module_prompt_rules(
    _MODULE_PROMPT_RULES_DIR, _MODULE_PROMPT_RULES_LEGACY_PATH
)


_TABLE_SELECTOR_DOMAIN_HINT = """You are selecting tables for the PROD module, which has three sub-areas:
- MFC (Manufacturing): items, routing, BOM, MBOM — manufacturing master/setup data.
- PRD (Production): sales orders, rework orders, scrap work orders — production execution documents.
- SEQ (Sequencing): sequencing / reference tied to a work order — tracks execution order against a work order.

Match the question to the sub-area it belongs to, then pick the table(s) from that sub-area's tables in the catalog below."""


_TABLE_SELECTOR_RULES_DIR = _DATA / "table_selector_rules"


def _load_table_selector_rules(dir_path: Path) -> Dict[str, str]:
    """
    Loads the dedicated table-selection catalog/rule text used ONLY by
    resolve_tables_llm() (Call B / the table-selector model, currently
    XiYanSQL-7B on CPU). Each file under `dir_path` is a JSON object
    keyed by module_name -> rule text (list of lines, joined with "\\n",
    or a plain string) — e.g. {"prod": ["...", "..."]}. A file's own stem
    (minus an optional "_table_selector_rules" suffix) is also registered
    as a fallback key, so a file with no internal module keys still works.

    Returns an empty dict (never raises) if `dir_path` doesn't exist yet
    or a file fails to parse — callers must fall back to
    _TABLE_SELECTOR_DOMAIN_HINT in that case; see _get_table_selector_rule_text().
    """
    _SUFFIX = "_table_selector_rules"
    rules: Dict[str, str] = {}
    if not dir_path.is_dir():
        log.warning(
            "[startup] %s not found — table selector will use the generic "
            "_TABLE_SELECTOR_DOMAIN_HINT for every module. Add a "
            "<module>_table_selector_rules.json file there to give the "
            "table-selector model its own dedicated catalog/hierarchy text.",
            dir_path,
        )
        return rules
    for path in sorted(dir_path.glob("*.json")):
        stem = path.stem
        file_key = stem[: -len(_SUFFIX)] if stem.lower().endswith(_SUFFIX) else stem
        file_key = file_key.lower()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    module_key = str(k).strip().lower()
                    if module_key in rules:
                        log.warning(
                            "[startup] Module key %r from %s overwrites a "
                            "table-selector rule already loaded from another "
                            "file — module keys must be unique.",
                            module_key, path,
                        )
                    rules[module_key] = _parse_rule_value(v)
            else:
                rules[file_key] = _parse_rule_value(raw)
        except Exception as exc:
            log.error(
                "[startup] Failed to load table-selector rules from %s: %s — skipping",
                path, exc,
            )
    if rules:
        log.info(
            "[startup] Loaded dedicated table-selection catalog for module(s): %s (from %s)",
            list(rules.keys()), dir_path,
        )
    return rules


TABLE_SELECTOR_PROMPT_RULES: Dict[str, str] = _load_table_selector_rules(_TABLE_SELECTOR_RULES_DIR)


def _get_table_selector_rule_text(module_name: str) -> str:
    """Look up the dedicated table-selection rule text for `module_name`,
    falling back to the generic _TABLE_SELECTOR_DOMAIN_HINT if this module
    has no data/table_selector_rules/*.json entry of its own."""
    key = (module_name or "").strip().lower()
    return TABLE_SELECTOR_PROMPT_RULES.get(key, _TABLE_SELECTOR_DOMAIN_HINT)


_FILE_KEY_TO_SUBMODULES: Dict[str, List[str]] = {}
for _mk, _fk in _MODULE_TO_FILE_KEY.items():
    if _mk == _fk:
        continue  # the container-level combined key itself, not a real submodule
    _FILE_KEY_TO_SUBMODULES.setdefault(_fk, []).append(_mk)
_FILE_KEY_TO_SUBMODULES = {k: v for k, v in _FILE_KEY_TO_SUBMODULES.items() if len(v) >= 2}
if _FILE_KEY_TO_SUBMODULES:
    log.info(
        "[startup] Submodule pre-classification available for container(s): %s",
        {k: len(v) for k, v in _FILE_KEY_TO_SUBMODULES.items()},
    )


def _get_module_prompt_rules(module_name: str) -> str:
    """
    Fetch the rule block for the given real module (items/bom/routing/...),
    with that module's container file's cross-module bridging/hierarchy
    text (the "_global_hierarchy" key, e.g. how MFC/PRD/SEQ interconnect)
    automatically prepended, if that container defines one. This is what
    lets an "items" question still carry PRD/SEQ bridging context, a
    "pickslip" question still carry the MFC item-master bridging context,
    etc. — without hand-copying the bridging text into every module block.
    """
    if not module_name:
        return ""
    key = module_name.strip().lower()
    own_text = MODULE_PROMPT_RULES.get(key, "")
    if not own_text:
        return ""
    file_key = _MODULE_TO_FILE_KEY.get(key)
    global_text = _GLOBAL_RULES_BY_FILE_KEY.get(file_key, "") if file_key else ""
    if global_text:
        return f"{global_text}\n\n{own_text}"
    return own_text



_CONCEPT_RULE_LINE_RE = re.compile(r"^-\s*(.+?)\s*(?:\u2192|->)\s*(.+)$")

_CONCEPT_TABLE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")


def _derive_concept_rules_from_text(rule_text: str) -> List[Tuple[str, List[str]]]:
    """
    Auto-derives (concept_description, [required_table, ...]) pairs
    directly from a module's existing free-text prompt rules — the same
    "- <when this applies> → <TABLE_NAME>" bullets already hand-written in
    data/module_prompt_rules/*.json to guide the final table selector
    (Call B). No separate concept_table_map needs to be authored anywhere;
    every concept the system can narrow on is whatever is already written
    in that module's own rule text, re-parsed fresh every time rules load.

    A line qualifies as a concept rule when it:
      - starts with "-" (a bullet), AND
      - contains a "→" (or "->") arrow, AND
      - the arrow's target — BEFORE any parenthetical aside — contains at
        least one ALLCAPS_WITH_UNDERSCORES token.

    Parenthetical asides on the target side (e.g. "(bridge to
    MFC_M_ROUTING_HEADER in ROUTING module)") are dropped before scanning
    for table tokens, since those typically name a table from a DIFFERENT
    module/subgroup for context — not one this subgroup's own candidate
    set should require. (resolve_tables_llm() separately drops any
    required table that isn't actually in the current subgroup, so a
    stray cross-module mention slipping through here is still harmless.)

    Multiple tables on the target side (joined by "or", commas, etc.) are
    all kept as alternatives — narrowing only needs to produce a safe
    candidate superset; Call B still picks the actually-correct table(s)
    out of it, exactly as it does today.

    Bullets with no arrow (general notes like "Never pick a table not
    listed above.") and modules that only use prose "RULE:" paragraphs
    (no arrow-bullets at all) simply produce an empty list — for those,
    concept-narrowing is skipped and Call B falls back to the full
    subgroup catalog exactly as it did before this feature existed.
    """
    rules: List[Tuple[str, List[str]]] = []
    for raw_line in rule_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        m = _CONCEPT_RULE_LINE_RE.match(line)
        if not m:
            continue
        concept_text = m.group(1).strip().strip('"').strip()
        target_head = m.group(2).split("(", 1)[0]  # drop parenthetical asides
        tables = list(dict.fromkeys(_CONCEPT_TABLE_TOKEN_RE.findall(target_head)))
        if concept_text and tables:
            rules.append((concept_text, tables))
    return rules



MODULE_CONCEPT_TABLE_MAP: Dict[str, List[Tuple[str, List[str]]]] = {
    module_key: _derive_concept_rules_from_text(rule_text)
    for module_key, rule_text in MODULE_PROMPT_RULES.items()
}
_module_concept_counts = {k: len(v) for k, v in MODULE_CONCEPT_TABLE_MAP.items() if v}
if _module_concept_counts:
    log.info(
        "[startup] Auto-derived concept rules for %d module(s): %s",
        len(_module_concept_counts), _module_concept_counts,
    )


def _get_concept_table_map(module_name: str) -> List[Tuple[str, List[str]]]:
    """
    Fetch [(concept_description, [required tables]), ...] for the given
    real module. Returns [] for modules whose rule text has no
    arrow-bullet concept rules — callers must treat that as
    "concept-narrowing not applicable to this module" and fall back to
    the existing full-subgroup-catalog flow, not as an error.
    """
    if not module_name:
        return []
    return MODULE_CONCEPT_TABLE_MAP.get(module_name.strip().lower(), [])

def _build_meta_from_module_data(module_data: dict) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for mod_val in module_data.get("modules", {}).values():
        for sg_val in mod_val.get("subgroups", {}).values():
            sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
            for tbl in sg_list:
                tname = tbl.get("table_name", "")
                if tname:
                    meta[tname] = tbl
    return meta

META: Dict[str, dict] = _build_meta_from_module_data(MODULE_DATA)

# ============================================================
# SCHEMA CACHES — loaded from build_schema_cache.py outputs
# ============================================================
_FULL_SCHEMA_CACHE_PATH = Path(os.getenv("FULL_SCHEMA_CACHE_PATH", str(_DATA / "full_schema_cache.json")))
_PROMPT_SCHEMA_CACHE_PATH = Path(os.getenv("PROMPT_SCHEMA_CACHE_PATH", str(_DATA / "prompt_schema_cache.json")))
_AUDIT_COL_SUFFIXES: tuple = tuple(
    s.strip()
    for s in os.getenv("AUDIT_COL_SUFFIXES", "cdate,cuser,udate,uuser,tenant_id,notes,remarks,description").split(",")
    if s.strip()
)

def _derive_prompt_schema_from_full(full: Dict[str, dict]) -> Dict[str, dict]:
    """Strip audit columns from full schema; fallback to suffix matching if audit_columns absent."""
    prompt: Dict[str, dict] = {}
    for tbl, info in full.items():
        all_cols = info.get("columns", [])
        audit_set = {c.lower() for c in info.get("audit_columns", [])}
        if not audit_set:
            audit_set = {
                c.lower() for c in all_cols
                if any(c.lower().endswith(s) for s in _AUDIT_COL_SUFFIXES)
            }
        non_audit = [c for c in all_cols if c.lower() not in audit_set]
        prompt[tbl] = {
            "columns": non_audit,
            "pk": info.get("pk", []),
            "fk": info.get("fk", []),
            "date_columns": info.get("date_columns", []),
            "status_columns": info.get("status_columns", []),
            "business_columns": info.get("business_columns", []),
            "join_paths": info.get("join_paths", {}),
        }
    return prompt

def load_schema_caches() -> tuple:
    """Load full_schema_cache.json (required). PROMPT_SCHEMA_CACHE is derived from it."""
    if not _FULL_SCHEMA_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Schema cache file not found: {_FULL_SCHEMA_CACHE_PATH}. "
            "Run build_schema_cache.py to generate it."
        )
    full = json.loads(_FULL_SCHEMA_CACHE_PATH.read_text(encoding="utf-8"))
    prompt = _derive_prompt_schema_from_full(full)
    log.info(
        "[schema_cache] Loaded full_schema_cache: %d tables | "
        "PROMPT_SCHEMA_CACHE derived: %d tables (audit cols stripped)",
        len(full), len(prompt),
    )
    return full, prompt

FULL_SCHEMA_CACHE: Dict[str, dict] = {}
PROMPT_SCHEMA_CACHE: Dict[str, dict] = {}

try:
    FULL_SCHEMA_CACHE, PROMPT_SCHEMA_CACHE = load_schema_caches()
    log.info("[schema_cache] Schema cache loaded and prompt schema derived successfully.")
except FileNotFoundError as _sce:
    log.warning(
        "[schema_cache] full_schema_cache.json missing - run build_schema_cache.py. %s", _sce
    )

# ── Schema-enrichment config ───────────────────────────────────────────────
_SCHEMA_COL_MAX: int = int(os.getenv("SCHEMA_COL_MAX", "7").strip())

def _get_table_pk_cols(table: str) -> set:
    """
    Return lowercased PK column names for a table.
    Sources: META is_pk flag → FULL_SCHEMA_CACHE pk list → COL_ROLE_INDEX identifier role.
    """
    pk: set = set()
    for col, cinfo in META.get(table, {}).get("columns", {}).items():
        if isinstance(cinfo, dict) and cinfo.get("is_pk"):
            pk.add(col.lower())
    for col in FULL_SCHEMA_CACHE.get(table, {}).get("pk", []):
        pk.add(col.lower())
    for col, roles in COL_ROLE_INDEX.get(table, {}).items():
        if "identifier" in roles:
            pk.add(col.lower())
    return pk

def _strip_pk_columns(result: List[dict], tables: List[str]) -> List[dict]:
    """
    Remove primary-key columns from the final result set before it reaches
    the user. PKs are often required in SQL for GROUP BY/JOIN correctness
    but were never asked for and shouldn't be shown in the answer.
    Keeps all columns if stripping would leave zero columns in a row
    (e.g. a lookup query that only ever selected the PK itself).
    """
    if not result:
        return result

    pk_cols: set = set()
    for t in tables:
        pk_cols |= _get_table_pk_cols(t)
        pk_cols |= _get_table_pk_cols(t.upper())

    if not pk_cols:
        return result

    row_keys = list(result[0].keys())
    keep_keys = [k for k in row_keys if k.lower() not in pk_cols]

    if not keep_keys:
        return result

    return [{k: row[k] for k in keep_keys} for row in result]

def _has_pk_only_group_by(sql: str) -> bool:
    """
    Detects the "total turned into per-row count" bug: an aggregate
    (COUNT/SUM/...) combined with a GROUP BY whose ONLY column(s) are the
    primary key of the table being grouped. A PK is unique per row, so
    grouping by it always yields exactly one row per record — silently
    turning what should be a single scalar total ("total number of items")
    into N rows each showing e.g. count=1. A legitimate "group by category"
    query groups by a real business dimension (status, vendor, type), never
    by the row's own identity column, so this has no false positives
    against genuine per-category breakdowns.

    Mirrors gpu_service.py::_has_pk_only_group_by (same bug, same fix) —
    this copy exists as defense-in-depth for SQL that reaches this point
    without having gone through that check (e.g. GPU retries exhausted and
    "proceeded anyway", or the bare column got added locally by
    enforce_mandatory_select_columns + _fix_group_by_coherence above).
    """
    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
        sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        if not group:
            return False
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return False
        if not any(sel_expr.find(*_AGG_TYPES) for sel_expr in select.expressions):
            return False

        group_cols = list(group.find_all(sqlglot.exp.Column))
        if not group_cols:
            return False
        ctx = _SqlResolutionContext(expr, [])
        for gcol in group_cols:
            tbl = gcol.table or ctx.resolve_owner(gcol.name) or ""
            col_name = (gcol.name or "").lower()
            if col_name not in _get_table_pk_cols(tbl) and col_name not in _get_table_pk_cols(tbl.upper()):
                return False
        return True
    except Exception:
        return False


def _collapse_pk_group_by_to_scalar(sql: str) -> str:
    """Companion to _has_pk_only_group_by — strips the GROUP BY and drops
    every non-aggregated SELECT column, keeping only the aggregate
    expression(s) (and their aliases), collapsing back to a scalar total."""
    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
        sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql

        kept = [sel_expr for sel_expr in select.expressions if sel_expr.find(*_AGG_TYPES)]
        if not kept:
            return sql
        select.set("expressions", kept)
        expr.set("group", None)
        result = expr.sql(dialect="tsql")
        log.info("[pipeline] Collapsed spurious PK GROUP BY to scalar total: %s", result)
        return result
    except Exception as exc:
        log.warning("[pipeline] Failed to collapse PK GROUP BY (%s) — leaving SQL as-is", exc)
        return sql


def _has_spurious_identifier_in_group_by(sql: str) -> bool:
    """
    Sibling to _has_pk_only_group_by, for the MIXED case it deliberately
    doesn't cover: a GROUP BY that combines a real business dimension
    (e.g. Item_Type_VSP_ID) with an identifier/PK column dragged in from a
    JOINed table (e.g. Item_Attributes_ID from a link/attribute table).

    _has_pk_only_group_by only fires when EVERY group-by column is a PK,
    so a query like

        SELECT T.Type, COUNT(T.Item_ID), J.Attr_ID
        FROM T JOIN J ON T.Item_ID = J.Item_ID
        GROUP BY T.Type, J.Attr_ID

    slips through it untouched — T.Type is a real dimension, so that check
    returns False on the first column and never inspects J.Attr_ID. But
    J.Attr_ID is still unique (or near-unique) per joined row, so it
    shatters each T.Type bucket back down to ~1 row per join match: the
    same silent "total became per-row count" failure as the pure-PK case,
    just partial instead of total, and easy to miss because the output
    still looks like a legitimate per-type breakdown.

    Returns True iff at least one GROUP BY column is a PK/identifier
    column AND at least one is not (i.e. a real dimension is present, but
    so is a spurious identity column that shouldn't be there). Mirrors
    gpu_service.py::_has_spurious_identifier_in_group_by.
    """
    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
        sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        if not group:
            return False
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return False
        if not any(sel_expr.find(*_AGG_TYPES) for sel_expr in select.expressions):
            return False

        group_cols = list(group.find_all(sqlglot.exp.Column))
        if not group_cols:
            return False

        pk_flags = []
        ctx = _SqlResolutionContext(expr, [])
        for gcol in group_cols:
            tbl = gcol.table or ctx.resolve_owner(gcol.name) or ""
            col_name = (gcol.name or "").lower()
            is_pk = col_name in _get_table_pk_cols(tbl) or col_name in _get_table_pk_cols(tbl.upper())
            pk_flags.append(is_pk)
        return any(pk_flags) and not all(pk_flags)
    except Exception:
        return False


def _strip_identifier_cols_from_group_by(sql: str) -> str:
    """
    Companion to _has_spurious_identifier_in_group_by. Removes only the
    spurious PK/identifier column(s) from GROUP BY — keeping real
    dimension columns intact — drops matching bare (non-aggregated)
    columns from SELECT so the result stays valid SQL, and, if a JOIN is
    present, switches COUNT(col) to COUNT(DISTINCT col) so the join's row
    fan-out can't inflate the total either.
    """
    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
        sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        select = expr.find(sqlglot.exp.Select)
        if not group or not select:
            return sql

        keep_group, drop_names = [], set()
        ctx = _SqlResolutionContext(expr, [])
        for gcol in group.find_all(sqlglot.exp.Column):
            was_bare = not gcol.table
            tbl = gcol.table or ctx.resolve_owner(gcol.name) or ""
            col_name = (gcol.name or "").lower()
            is_pk = col_name in _get_table_pk_cols(tbl) or col_name in _get_table_pk_cols(tbl.upper())
            if is_pk:
                drop_names.add((tbl.lower(), col_name))
                if was_bare:
                    drop_names.add(("", col_name))
            else:
                keep_group.append(gcol)

        if not keep_group:
            return sql
        group.set("expressions", keep_group)

        kept_select = []
        for sel_expr in select.expressions:
            col = sel_expr if isinstance(sel_expr, sqlglot.exp.Column) else sel_expr.find(sqlglot.exp.Column)
            is_bare_dropped_col = (
                col is not None
                and not sel_expr.find(*_AGG_TYPES)
                and ((col.table or "").lower(), (col.name or "").lower()) in drop_names
            )
            if is_bare_dropped_col:
                continue
            kept_select.append(sel_expr)
        if kept_select:
            select.set("expressions", kept_select)
        if expr.find(sqlglot.exp.Join):
            for sel_expr in select.expressions:
                cnt = sel_expr.find(sqlglot.exp.Count)
                if cnt is not None and not isinstance(cnt.this, sqlglot.exp.Distinct):
                    cnt.set("this", sqlglot.exp.Distinct(expressions=[cnt.this]))

        result = expr.sql(dialect="tsql")
        log.info("[pipeline] Stripped spurious identifier column(s) from mixed GROUP BY: %s", result)
        return result
    except Exception as exc:
        log.warning("[pipeline] Failed to strip spurious identifier GROUP BY cols (%s) — leaving SQL as-is", exc)
        return sql

_GROUP_BY_DIMENSION_RE = re.compile(
    r'\b(?:per|for\s+each|each)\s+([a-zA-Z][a-zA-Z_]{2,})\b',
    re.IGNORECASE,
)

_GROUP_BY_BY_RE = re.compile(
    r'(?<!sort )(?<!sorted )(?<!order )(?<!ordered )(?<!filter )(?<!filtered )'
    r'\bby\s+([a-zA-Z][a-zA-Z_]{2,})\b',
    re.IGNORECASE,
)
_GROUP_BY_HINT_STOPWORDS = {
    "the", "a", "an", "this", "that", "each", "date", "day", "week",
    "month", "year", "default", "far", "now",
}


def _extract_group_by_dimension_hints(question: str) -> List[str]:
    """
    Pull candidate dimension nouns out of explicit per-group phrasing
    ("per item", "for each vendor", "each plant", "breakdown by supplier").
    Returns [] for plain questions with no such phrasing ("total received
    quantity", "how many receipts") so those are never forced into a
    GROUP BY they didn't ask for.
    """
    hints: List[str] = []
    for pattern in (_GROUP_BY_DIMENSION_RE, _GROUP_BY_BY_RE):
        for raw in pattern.findall(question or ""):
            word = raw.lower().strip()
            if word in _GROUP_BY_HINT_STOPWORDS or word in hints:
                continue
            hints.append(word)
    return hints


def _resolve_group_by_dimension_col(
    hint: str, table: str, live_cols: Dict[str, List[str]]
) -> Optional[str]:
    """
    Match a dimension hint word ("item", "supplier", "vendor") from the
    question against this table's live/prompt columns. Prefers an exact
    "<hint>_id" / "<hint>_code" / "<hint>_number" / "<hint>_name" match,
    then a column that starts with the hint, then any column containing
    it. Never returns one of the table's own PK columns — grouping by the
    row's own identity column is never a legitimate per-X breakdown (see
    _has_pk_only_group_by), it's the exact bug this is guarding against.
    """
    cols = live_cols.get(table, [])
    if not cols:
        return None

    pk_cols = _get_table_pk_cols(table) | _get_table_pk_cols(table.upper())
    hint_l = hint.lower()
    if hint_l.endswith("s") and len(hint_l) > 3 and not hint_l.endswith("ss"):
        hint_l = hint_l[:-1] 

    candidates = [c for c in cols if c.lower() not in pk_cols]
    if not candidates:
        return None

    for suffix in ("_id", "_code", "_number", "_name"):
        target = f"{hint_l}{suffix}"
        for c in candidates:
            if c.lower() == target:
                return c
    for c in candidates:
        if c.lower() == hint_l or c.lower().startswith(hint_l + "_"):
            return c
    for c in candidates:
        if hint_l in c.lower():
            return c
    return None


def _fix_missing_group_by_dimension(
    sql: str,
    question: str,
    intent: str,
    llm_tables: List[str],
    live_cols: Dict[str, List[str]],
) -> str:
    """
    Safety net for the "per item" bug: the question explicitly asks for a
    per-dimension breakdown ("list the received quantity per item") but
    the LLM emitted a plain scalar aggregate with no such GROUP BY at all.
    Left alone, that falls through to enforce_mandatory_select_columns
    injecting the table's PK into SELECT, _fix_group_by_coherence grouping
    by that bare PK, and _has_pk_only_group_by collapsing it right back to
    the same scalar total the user didn't ask for — the per-item breakdown
    never actually happens.

    Only fires when ALL of:
      - intent is an aggregation-style intent (aggregation / group_by /
        relational_aggregation) — never touches simple/lookup/existence
        questions, so a plain "total X" or "show me X" is untouched
      - the question contains explicit per-group phrasing ("per X",
        "for each X", "each X", or safely-scoped "by X") — a question
        with no such phrasing yields no hints and this is a no-op
      - the SQL already has an aggregate function
      - the SQL's current GROUP BY (if any) is empty or PK-only — i.e.
        there is no real dimension being grouped on yet, so this never
        overrides a GROUP BY the model already got right
      - a hint from the question resolves to an actual non-PK column on
        one of the candidate tables

    When those hold, the resolved dimension column becomes the query's
    GROUP BY (and is added to SELECT if missing), replacing any PK-only
    GROUP BY outright so the downstream PK-collapse guard has nothing
    left to undo.
    """
    _AGG_INTENTS = {"aggregation", "group_by", "relational_aggregation"}
    if intent not in _AGG_INTENTS or not sql or not llm_tables:
        return sql

    hints = _extract_group_by_dimension_hints(question)
    if not hints:
        return sql

    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
        sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql
        if not any(sel_expr.find(*_AGG_TYPES) for sel_expr in select.expressions):
            return sql # no aggregate — nothing for a GROUP BY to attach to

        group = expr.find(sqlglot.exp.Group)
        if group is not None:
            group_cols = list(group.find_all(sqlglot.exp.Column))
            if group_cols:
                _ctx = _SqlResolutionContext(expr, [])
                all_pk = all(
                    (gc.name or "").lower() in (
                        _get_table_pk_cols(gc.table or _ctx.resolve_owner(gc.name) or "")
                        | _get_table_pk_cols((gc.table or _ctx.resolve_owner(gc.name) or "").upper())
                    )
                    for gc in group_cols
                )
                if not all_pk:
                    return sql 

        dim_table, dim_col = None, None
        for hint in hints:
            for tbl in llm_tables:
                col = _resolve_group_by_dimension_col(hint, tbl, live_cols)
                if col:
                    dim_table, dim_col = tbl, col
                    break
            if dim_col:
                break

        if not dim_col:
            return sql

      
        _qual_ctx = _SqlResolutionContext(expr, llm_tables, live_cols=live_cols, scope_select=select)
        dim_qualifier = _qual_ctx.qualifier_for(dim_table.upper())

        new_col = sqlglot.exp.Column(
            this=sqlglot.exp.Identifier(this=dim_col),
            table=sqlglot.exp.Identifier(this=dim_qualifier),
        )

        already_selected = any(
            isinstance(c, sqlglot.exp.Column)
            and (c.name or "").lower() == dim_col.lower()
            and (c.table or "").lower() in ("", dim_table.lower(), dim_qualifier.lower())
            for c in select.find_all(sqlglot.exp.Column)
        )
        if not already_selected:
            select.append("expressions", new_col.copy())

        expr.set("group", sqlglot.exp.Group(expressions=[new_col.copy()]))

        result = expr.sql(dialect="tsql")
        log.info(
            "[fix_missing_group_by_dimension] Question implies per-%s breakdown — "
            "grouping by %s.%s instead of a scalar total",
            hints[0], dim_qualifier, dim_col,
        )
        _vprint(
            "SCHEMA", "cyan",
            f" Synthesized GROUP BY {dim_qualifier}.{dim_col} from 'per {hints[0]}' phrasing",
        )
        return result
    except Exception as exc:
        log.warning("[fix_missing_group_by_dimension] Failed: %s", exc)
        return sql


def _get_join_fk_cols(join_clause: Optional[str]) -> set:
    """Extract lowercased column names from a JOIN ... ON clause. No suffix guessing."""
    if not join_clause:
        return set()
    return {m.lower() for m in re.findall(r"\b\w+\.(\w+)\b", join_clause)}

SCHEMA: Dict[str, dict] = {
    tbl: {col: {} for col in info.get("columns", [])}
    for tbl, info in FULL_SCHEMA_CACHE.items()
} if FULL_SCHEMA_CACHE else {}

# ============================================================
# COLUMN ROLE INDEX — built from module narrow JSON at startup
# ============================================================
_COL_DESC_ROLE_SIGNALS: Dict[str, List[str]] = {
    "identifier": [
        "unique system-generated id", "unique record id", "unique id",
        "unique identifier", "primary key", "system-generated id",
    ],
    "name": [
        "name", "code", "title", "label", "description", "designation",
    ],
    "quantity": [
        "quantity", "qty", "count", "total", "amount", "weight",
        "volume", "length", "duration", "number of",
    ],
    "date": [
        "date", "timestamp", "datetime", "validity", "effective",
        "expiry", "due date", "start date", "end date",
    ],
    "status": [
        "status", "state", "flag", "indicator",
        "active", "inactive", "enabled", "disabled",
    ],
    "foreign_key": [
        "reference to", "foreign key", "refers to",
        "links to", "mapped to", "associated with",
    ],
    "audit": [
        "user who created", "user who updated", "user who last",
        "created by", "updated by", "tenant", "organization identifier",
    ],
}

def _classify_col_desc(description: str) -> List[str]:
    dl = description.lower()
    return [
        role
        for role, signals in _COL_DESC_ROLE_SIGNALS.items()
        if any(sig in dl for sig in signals)
    ]

def _build_col_role_index(module_data: dict) -> Dict[str, Dict[str, List[str]]]:
    index: Dict[str, Dict[str, List[str]]] = {}
    for mod_val in module_data.get("modules", {}).values():
        for sg_val in mod_val.get("subgroups", {}).values():
            sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
            for tbl in sg_list:
                tname: str = tbl.get("table_name", "")
                if not tname:
                    continue
                col_roles: Dict[str, List[str]] = {}
                explicit_time = {c.lower(): c for c in tbl.get("time_columns", [])}
                explicit_value = {c.lower(): c for c in tbl.get("value_columns", [])}
                explicit_qty = {c.lower(): c for c in tbl.get("quantity_columns", [])}
                explicit_status = {c.lower(): c for c in tbl.get("status_columns", [])}
                explicit_fk = {c.lower(): c for c in tbl.get("foreign_keys", [])}
                for col, desc in tbl.get("columns", {}).items():
                    cl = col.lower()
                    roles: List[str] = []
                    if cl in explicit_time:
                        roles.append("date")
                    if cl in explicit_qty:
                        roles.append("quantity")
                    if cl in explicit_value:
                        roles.append("quantity")
                    if cl in explicit_status:
                        roles.append("status")
                    if cl in explicit_fk:
                        roles.append("foreign_key")
                    if desc:
                        desc_roles = _classify_col_desc(desc)
                        for r in desc_roles:
                            if r not in roles:
                                roles.append(r)
                    if roles:
                        col_roles[col] = roles
                index[tname] = col_roles
    log.info("[col_role_index] Built column role index: %d tables, %d total column entries",
             len(index), sum(len(v) for v in index.values()))
    return index

COL_ROLE_INDEX: Dict[str, Dict[str, List[str]]] = _build_col_role_index(MODULE_DATA)

def _build_narrow_table_index(module_data: dict) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for mod_val in module_data.get("modules", {}).values():
        for sg_val in mod_val.get("subgroups", {}).values():
            sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
            for tbl in sg_list:
                tname = tbl.get("table_name", "")
                if tname:
                    index[tname] = tbl
    return index

NARROW_TABLE_INDEX: Dict[str, dict] = _build_narrow_table_index(MODULE_DATA)

USER_TIMEZONE: str = os.getenv("USER_TIMEZONE", "Asia/Kolkata").strip()
TZ = pytz.timezone(USER_TIMEZONE)

_LLM_SERVER_URL: str = os.getenv(
    "LLM_SERVER_URL",
    "http://localhost:18081/completion"
).strip()

_LLM_TIMEOUT: int = int(
    os.getenv("LLM_TIMEOUT_SECS", "180").strip()
)
_OLLAMA_MODEL: str = os.getenv(
    "OLLAMA_MODEL",
    "xiyansql7b:latest"
).strip()

_DATE_PRIORITY_SUFFIXES: List[str] = [
    s.strip()
    for s in os.getenv("DATE_PRIORITY_SUFFIXES", "created_date,cdate,start_date,end_date,modified_date").split(",")
    if s.strip()
]

_AUDIT_COL_SUFFIXES: tuple = tuple(
    s.strip()
    for s in os.getenv("AUDIT_COL_SUFFIXES", "cdate,cuser,udate,uuser,tenant_id,notes,remarks,description").split(",")
    if s.strip()
)

_LOG_SUFFIXES: tuple = ("_LOG", "_HISTORY", "_TYPE", "_AUDIT", "_ARCHIVE")

_DEFAULT_TOP: int = int(os.getenv("DEFAULT_TOP", "50").strip())
_MAX_QUESTION_LEN: int = int(os.getenv("MAX_QUESTION_LEN", "1000").strip())
_MAX_RESPONSE_ROWS: int = int(os.getenv("MAX_RESPONSE_ROWS", "50").strip())


_CHAT_HISTORY_MAX_ROWS: int = int(os.getenv("CHAT_HISTORY_MAX_ROWS", str(_MAX_RESPONSE_ROWS)).strip())


_SQL_INJECTION_PATTERNS: tuple = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        r"\bSELECT\b[\s\S]{0,200}\bFROM\b",            # SELECT ... FROM
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\s+[\w\.\[\]]+\s+SET\b",
        r"\bDELETE\s+FROM\b",
        r"\bDROP\s+(TABLE|DATABASE|VIEW|INDEX|PROCEDURE|FUNCTION)\b",
        r"\bALTER\s+(TABLE|DATABASE|VIEW)\b",
        r"\bTRUNCATE\s+TABLE\b",
        r"\bCREATE\s+(TABLE|DATABASE|VIEW|INDEX|PROCEDURE|FUNCTION)\b",
        r"\bEXEC(?:UTE)?\s*\(?\s*[\w\.\[\]#]+",
        r"\bUNION\s+(ALL\s+)?SELECT\b",
        r"\bSHOW\s+(TABLES|DATABASES|COLUMNS)\b",
        r"^\s*DESC(?:RIBE)?\s+[A-Za-z_][\w\.\[\]]*\s*;?\s*$",  # SQL DESCRIBE <table>, not English "describe ..."
        r"\bMERGE\s+INTO\b",
        r"\bGRANT\s+\w+\s+ON\b",
        r"\bREVOKE\s+\w+\s+ON\b",
        r"\bWAITFOR\s+DELAY\b",
        r"\bxp_cmdshell\b",
        r"\bsp_executesql\b",
        r"\binformation_schema\.\w+",
        r"\bsys\.(tables|columns|databases|objects)\b",
        r";\s*--",                                      # statement terminator + comment
        r"--\s*$",                                       # trailing SQL comment
        r"/\*[\s\S]*?\*/",                                # block comment
        r"'\s*OR\s*'?\s*1\s*'?\s*=\s*'?\s*1",             # ' OR 1=1 / ' OR '1'='1
        r"'\s*;\s*\w",                                    # quote-terminated stacked statement
    )
)


def _looks_like_raw_sql(question: str) -> bool:
    """True if `question` appears to be a raw SQL statement or an
    injection payload rather than a natural-language question."""
    return any(p.search(question) for p in _SQL_INJECTION_PATTERNS)

_REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost").strip()
_REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379").strip())
_REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "").strip()
_REDIS_DB: int = int(os.getenv("REDIS_DB", "0").strip())
_REDIS_TTL: int = int(os.getenv("REDIS_TTL_SECS", "86400").strip())
_CACHE_SIM_THRESHOLD: float = float(os.getenv("CACHE_SIM_THRESHOLD", "0.92").strip())
_EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2").strip()
_CACHE_QDRANT_COLLECTION: str = os.getenv("CACHE_QDRANT_COLLECTION", "nl2sql_cache").strip()
_CACHE_EMBED_DIM: int = int(os.getenv("CACHE_EMBED_DIM", "384").strip())

_LOV_FUZZY_THRESHOLD: float = float(os.getenv("LOV_FUZZY_THRESHOLD", "0.82").strip())

_ALLOWED_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

_CORS_STRICT_MODE: bool = os.getenv("CORS_STRICT_MODE", "true").strip().lower() == "true"
_METRICS_TOKEN: str = os.getenv("METRICS_TOKEN", "").strip()

def _verify_metrics_token(request: Request) -> None:
    if not _METRICS_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token on /metrics")
    token = auth[len("Bearer "):]
    if not secrets.compare_digest(token, _METRICS_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid metrics token")

def _validate_cors_origins(origins: List[str], strict: bool) -> None:
    if not origins:
        msg = (
            "ALLOWED_ORIGINS is not set — every browser request will be blocked by CORS. "
            "Set ALLOWED_ORIGINS=https://your-frontend.example.com in .env. "
            "To downgrade this to a warning, set CORS_STRICT_MODE=false."
        )
        if strict:
            raise RuntimeError(f"[CORS] {msg}")
        log.warning("[cors] %s", msg, extra={"event": "cors_misconfigured"})
        return

    for origin in origins:
        if origin == "*":
            msg = (
                "Wildcard '*' in ALLOWED_ORIGINS exposes your API to any origin. "
                "Use explicit frontend URL(s) in production, or set CORS_STRICT_MODE=false."
            )
            if strict:
                raise RuntimeError(f"[CORS] {msg}")
            log.warning("[cors] %s", msg, extra={"event": "cors_wildcard", "origin": origin})

        if origin.endswith("/"):
            log.warning(
                "[cors] Origin '%s' has a trailing slash — browsers omit it, "
                "causing silent CORS failures. Remove the trailing slash.",
                origin,
                extra={"event": "cors_trailing_slash", "origin": origin},
            )

_validate_cors_origins(_ALLOWED_ORIGINS, _CORS_STRICT_MODE)

_API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "").strip()
if not _API_SECRET_KEY:
    raise RuntimeError(
        "API_SECRET_KEY is not set in .env — refusing to start with an unprotected API. "
        "Set a strong random value, e.g.: openssl rand -hex 32"
    )
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Session token ──────────────────────────────────
_SESSION_SECRET: str = os.getenv("SESSION_SECRET", secrets.token_hex(32))
_SESSION_TOKEN_TTL: int = int(os.getenv("SESSION_TOKEN_TTL_SECS", "3600"))

def _make_session_token() -> str:
    """Create a signed, time-stamped session token: <expires_ts>.<hmac>"""
    expires = int(time.time()) + _SESSION_TOKEN_TTL
    payload = f"{expires}"
    sig = hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"

def _validate_session_token(token: str) -> bool:
    """Return True if the token is valid and not expired."""
    try:
        expires_str, sig = token.split(".", 1)
        expires = int(expires_str)
        expected = hmac.new(_SESSION_SECRET.encode(), expires_str.encode(), hashlib.sha256).hexdigest()
        if not secrets.compare_digest(sig, expected):
            return False
        return time.time() < expires
    except Exception:
        return False

async def _verify_session_or_api_key(request: Request, key: str = Security(_api_key_header)) -> None:
    """Accept either a static API key (server-to-server) or a short-lived session token
    passed as Authorization: Bearer <token> (browser clients)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if _validate_session_token(token):
            return
        raise HTTPException(status_code=401, detail="Session token expired or invalid. Please refresh.")
    if key and secrets.compare_digest(key, _API_SECRET_KEY):
        return
    raise HTTPException(status_code=403, detail="Forbidden")

_QDRANT_URL: str = os.getenv("QDRANT_URL", "").strip()
_QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "").strip()

# ── LLM-based table selector config ──────────────────────────────────────────
_TABLE_SELECTOR_ENABLED: bool = (
    os.getenv("TABLE_SELECTOR_LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes")
)

_TABLE_SELECTOR_MODEL: str = os.getenv("TABLE_SELECTOR_MODEL", "xiyansql-7b").strip()
_TABLE_SELECTOR_TIMEOUT_SECS: float = float(os.getenv("TABLE_SELECTOR_TIMEOUT_SECS", "60"))
_TABLE_SELECTOR_MAX_TOKENS: int = int(os.getenv("TABLE_SELECTOR_MAX_TOKENS", "200"))

# ── Chat topic titles (ChatGPT-style sidebar labels) ───────────────────────
# One tiny best-effort call, reusing the same on-prem model/server as the
# table selector above (_LLM_SERVER_URL / _LLM_BACKEND) — no separate model
# to provision. Only fired once per conversation, for the first question.
_CHAT_TOPIC_TIMEOUT_SECS: float = float(os.getenv("CHAT_TOPIC_TIMEOUT_SECS", "6"))
_CHAT_TOPIC_MAX_TOKENS: int = int(os.getenv("CHAT_TOPIC_MAX_TOKENS", "8"))
_CHAT_TOPIC_MAX_CHARS: int = int(os.getenv("CHAT_TOPIC_MAX_CHARS", "40"))
_TABLE_SELECTOR_MAX_CATALOG_CHARS: int = int(os.getenv("TABLE_SELECTOR_MAX_CATALOG_CHARS", "6000"))
_TABLE_SELECTOR_MAX_COLS_HINT: int = int(os.getenv("TABLE_SELECTOR_MAX_COLS_HINT", "12"))

_TABLE_SELECTOR_MAX_TABLES: int = int(os.getenv("TABLE_SELECTOR_MAX_TABLES", "4"))

_REDIS_KEY_PREFIX: str = os.getenv("REDIS_KEY_PREFIX", "nl2sql").strip().rstrip(":")
_TABLE_NAME_PREFIXES: List[str] = _get_table_name_prefixes_loader()

# ── GPU SERVICE CONFIG ────────────────────────────────────────────────────────

_GPU_SERVICE_URL: str = os.getenv(
    "GPU_SERVICE_URL",
    "http://10.225.30.26:8103/generate-sql"
).strip()
_GPU_TIMEOUT: float = float(os.getenv("GPU_TIMEOUT_SECS", "180"))
_GPU_RETRY_ATTEMPTS: int = int(os.getenv("GPU_RETRY_ATTEMPTS", "2"))

_LLM_COMPLEX_INTENTS: set = {
    s.strip()
    for s in os.getenv(
        "LLM_COMPLEX_INTENTS",
        "aggregation,group_by,existence,sort_by,join,relational_lookup,relational_aggregation"
    ).split(",")
    if s.strip()
}

_COL_WORD_STOPWORDS_CFG: set = {
    s.strip().lower()
    for s in os.getenv(
        "COL_WORD_STOPWORDS",
        "id,lov,date,time,flag,num,no,ref,val,code,name,desc,the,and,for,col,row,key,set,log"
    ).split(",")
    if s.strip()
}

_UI_FILENAME: str = os.getenv("UI_FILENAME", "mes_chatbot_v2.html").strip()
_DEBUG_RESPONSES: bool = os.getenv("DEBUG_RESPONSES", "false").strip().lower() == "true"
_SHOW_SQL: bool = os.getenv("SHOW_SQL", "true").strip().lower() in ("1", "true", "yes")
_JOIN_MAX_BFS_DEPTH: int = int(os.getenv("JOIN_MAX_BFS_DEPTH", "3"))
_TABLE_NAME_HIT_WEIGHT: int = int(os.getenv("TABLE_NAME_HIT_WEIGHT", "5"))

_COL_WORKFLOW_LOV_SUFFIXES: tuple = tuple(
    s.strip()
    for s in os.getenv("COL_WORKFLOW_LOV_SUFFIXES", "_status,_lov_id,_lov").split(",")
    if s.strip()
)
_COL_WORKFLOW_TYPE_SUFFIXES: tuple = tuple(
    s.strip()
    for s in os.getenv("COL_WORKFLOW_TYPE_SUFFIXES", "_type_id,_type").split(",")
    if s.strip()
)
_NO_TOP_INTENTS: set = {
    s.strip()
    for s in os.getenv(
        "NO_TOP_INTENTS",
        "aggregation,group_by,existence,lookup,relational_aggregation"
    ).split(",")
    if s.strip()
}
_AUDIT_LOG_ENABLED: bool = os.getenv("AUDIT_LOG_ENABLED", "true").strip().lower() == "true"
_DB_THREAD_POOL_SIZE: int = int(os.getenv("DB_THREAD_POOL_SIZE", "4"))
_db_executor = ThreadPoolExecutor(max_workers=_DB_THREAD_POOL_SIZE)

async def _verify_api_key(request: Request, key: str = Security(_api_key_header)) -> None:
    """Kept for backward compatibility — delegates to the session-aware validator."""
    await _verify_session_or_api_key(request, key)

from config.db import get_db_connection as _original_get_db_connection, get_pool_status as _get_pool_status

_DB_RETRY_ATTEMPTS = int(os.getenv("DB_RETRY_ATTEMPTS", "3"))
_DB_RETRY_BACKOFF = float(os.getenv("DB_RETRY_BACKOFF", "0.5"))
_DB_QUERY_TIMEOUT: int = int(os.getenv("DB_QUERY_TIMEOUT", "30"))

try:
    import pyodbc as _pyodbc
    _DB_TRANSIENT_EXC: tuple = (_pyodbc.OperationalError, _pyodbc.InterfaceError)
except ImportError:
    _DB_TRANSIENT_EXC = (OSError,)

@retry(
    stop=stop_after_attempt(_DB_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=_DB_RETRY_BACKOFF, min=0.5, max=5),
    retry=retry_if_exception_type(_DB_TRANSIENT_EXC),
    reraise=True,
)
def _get_db_connection_with_retry():
    return _original_get_db_connection()

get_db_connection = _get_db_connection_with_retry

def _check_readonly_user():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT HAS_PERMS_BY_NAME(NULL, NULL, 'INSERT') AS can_insert,
                       HAS_PERMS_BY_NAME(NULL, NULL, 'UPDATE') AS can_update,
                       HAS_PERMS_BY_NAME(NULL, NULL, 'DELETE') AS can_delete
            """)
            row = cur.fetchone()
            if row and (row[0] or row[1] or row[2]):
                log.warning(
                    "[startup] Database user has write permissions (INSERT/UPDATE/DELETE). "
                    "For production safety, use a read-only account."
                )
            else:
                log.info("[startup] Database user is read-only.")
    except Exception as e:
        log.warning(f"[startup] Could not verify database permissions: {e}")

# ============================================================
# GPU SERVICE CLIENT
# ============================================================

async def _call_gpu_service(
    question: str,
    tables: List[str],
    join_clause: Optional[str],
    intent: str,
    module_name: str,
    subgroup_name: str,
    live_cols: Dict[str, List[str]],
    resolved_filters: Optional[Dict[str, str]] = None,
    lov_matches: Optional[Dict[str, Union[str, List[str]]]] = None,
    reverse_hits: Optional[List[dict]] = None,
    module_prompt_rules: str = "",
    status_cb: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Call the GPU service to generate SQL.
    
    The GPU service handles:
    - Loading module metadata from its local copy
    - Building the XiYanSQL prompt
    - Generating SQL with internal retries for:
      - Ambiguous columns
      - GROUP BY errors
      - SQL repair
      - Invalid column names
    
    Returns: (sql, intent_override, retry_count)
    """
    if not _llm_circuit_breaker.allow_request():
        log.warning("[gpu] Circuit breaker OPEN — skipping GPU call")
        if _METRICS_ENABLED:
            _METRIC_LLM_ERRORS.labels(reason="circuit_open").inc()
            _METRIC_LLM_CIRCUIT_STATE.set(1)
        return None, None, 0

    _t0 = time.perf_counter()

    
    if status_cb is not None:
        try:
            _health_url = _GPU_SERVICE_URL.replace("/generate-sql", "/health")
            async with aiohttp.ClientSession() as _hs:
                async with _hs.get(
                    _health_url, timeout=aiohttp.ClientTimeout(total=3)
                ) as _hresp:
                    if _hresp.status == 200:
                        _hdata = await _hresp.json()
                        _pending = int(_hdata.get("queue", {}).get("pending", 0) or 0)
                        if _pending > 0:
                            await status_cb(
                                "Almost there...\nYour request is waiting for its turn and will begin shortly."
                            )
        except Exception as _health_exc:
            log.debug("[gpu] queue-depth health check skipped: %s", _health_exc)

        await status_cb("Analyzing your data...")

    # Prepare the request payload
    payload = {
        "question": question,
        "tables": tables,
        "join_clause": join_clause,
        "intent": intent,
        "module_name": module_name,
        "subgroup_name": subgroup_name,
        "live_cols": live_cols,
        "resolved_filters": resolved_filters or {},
        "lov_matches": lov_matches or {},
        "reverse_hits": reverse_hits or [],
        "module_prompt_rules": module_prompt_rules or "",
        "config": {
            "timeout_secs": _GPU_TIMEOUT,
            "max_retries": _GPU_RETRY_ATTEMPTS,
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _GPU_SERVICE_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_GPU_TIMEOUT + 10),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    log.error("[gpu] Service returned HTTP %d: %s", resp.status, error_text)
                    _llm_circuit_breaker.record_failure()
                    if _METRICS_ENABLED:
                        _METRIC_LLM_ERRORS.labels(reason="http_error").inc()
                    return None, None, 0
                
                data = await resp.json()
                sql = data.get("sql", "").strip()
                intent_override = data.get("intent")
                retry_count = data.get("retry_count", 0)
                error = data.get("error")
                
                if error:
                    log.warning("[gpu] Service returned error: %s", error)
                    _llm_circuit_breaker.record_failure()
                    if _METRICS_ENABLED:
                        _METRIC_LLM_ERRORS.labels(reason="gpu_error").inc()
                    return None, None, retry_count
                
                if not sql:
                    log.warning("[gpu] Service returned empty SQL")
                    _llm_circuit_breaker.record_failure()
                    if _METRICS_ENABLED:
                        _METRIC_LLM_ERRORS.labels(reason="empty_sql").inc()
                    return None, None, retry_count
                
                # Success
                _llm_circuit_breaker.record_success()
                if _METRICS_ENABLED:
                    _METRIC_LLM_LATENCY.observe(time.perf_counter() - _t0)
                    _METRIC_LLM_CIRCUIT_STATE.set(0)
                
                log.info("[gpu] SQL generated in %.2fs (retries=%d)",
                         time.perf_counter() - _t0, retry_count)
                return sql, intent_override, retry_count
                
    except asyncio.TimeoutError:
        log.error("[gpu] Service timeout after %.0fs", _GPU_TIMEOUT)
        _llm_circuit_breaker.record_failure()
        if _METRICS_ENABLED:
            _METRIC_LLM_ERRORS.labels(reason="timeout").inc()
        return None, None, 0
    except Exception as e:
        log.error("[gpu] Service call failed: %s", e)
        _llm_circuit_breaker.record_failure()
        if _METRICS_ENABLED:
            _METRIC_LLM_ERRORS.labels(reason="exception").inc()
        return None, None, 0


# ── Core module imports ───────────────────────────────────────────────────────
from core.date_resolver import (
    inject_date_filter,
    inject_effective_filter as _dr_inject_effective,
    inject_time_filter as _dr_inject_time_filter,
    strip_llm_date_filters,
    parse_date_range as _parse_date_range,
    has_temporal_intent as _dr_has_temporal_intent,
    has_current_intent as _dr_has_current_intent,
    has_effective_intent as _dr_has_effective_intent,
)
from core.type_resolver import (
    load_type_mappings,
    get_type_filters_all,
    get_type_lov_context,
)
from core.literal_resolver import (
    extract_literal_value,
    get_fk_map,
    init as _literal_resolver_init,
)
from core.prompt_builder import build_prompt as _ext_build_prompt
from core.graph_join_resolver import GraphJoinResolver
from core.neo4j_helper import get_neighbor_tables as _neo4j_get_neighbor_tables

# ── Post-response translation layer ──────────────
from core.response_translator import (
    load_translation_cache,
    load_lov_map as _translator_load_lov_map,
    translate_rows,
    cache_status as _translation_cache_status,
)
from core.translation_scheduler import start_cache_refresh_scheduler

# ── Reverse-lookup index ───────────
_MASTER_CACHE_PATH = Path(os.getenv(
    "MASTER_TRANSLATION_CACHE_PATH",
    str(_PROJECT_ROOT / "data" / "master_translation_cache.json"),
))
_REVERSE_LOOKUP_INDEX: Dict[str, List[dict]] = {}
_reverse_lookup_lock = _threading.Lock()

def _build_reverse_lookup_index(cache_path: Path = _MASTER_CACHE_PATH) -> Dict[str, List[dict]]:
    """
    Build a label → [{pk_col, id, label}] index from master_translation_cache.json.

    Actual cache format (flat, keyed by column name):
    {
      "fk": { "ITEM_ID": {"1": "Hydraulic Pump Assembly", "2": "Control Valve", ...}, ... },
      "lov": { "LIFECYCLE_STATUS_LOV_ID": {"2": "active", ...}, ... },
      "type":{ "ACTIVITY_TYPE_ID": {"1": "Production Activity", ...}, ... }
    }
    The index maps label_lower → list of {pk_col, id, label}.

    IMPORTANT: the key is the TOKEN-JOINED form of the label (same tokenizer as
    _reverse_lookup_question_values) so that labels with dots/slashes (e.g.
    "Copper Strip 0.5mm" → tokens ["copper","strip","0","5mm"] → key
    "copper strip 0 5mm") match the token-join of the question phrase.
    The original label string is preserved in the entry for display/hint use.
    """

    _TOKEN_RE = re.compile(r"[a-zA-Z0-9''\-]+")

    index: Dict[str, List[dict]] = {}
    if not cache_path.exists():
        log.warning(
            "[reverse_lookup] master_translation_cache.json not found at %s — "
            "reverse-lookup disabled. Run build_master_cache.py to enable.",
            cache_path,
        )
        return index

    try:
        raw: dict = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[reverse_lookup] Failed to load master_translation_cache.json: %s", exc)
        return index
    for section_name in ("fk", "lov", "type"):
        section = raw.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for pk_col, id_to_label in section.items():
            if not isinstance(id_to_label, dict):
                continue
            for raw_key, raw_val in id_to_label.items():
                id_val: str
                label: str
                key_str = str(raw_key).strip()
                if key_str.startswith("{") and "description" in key_str:
                    try:
                        parsed = ast.literal_eval(key_str)
                        label = str(parsed.get("description", "")).strip()
                    except Exception:
                        continue
                    id_val = str(raw_val).strip()
                elif isinstance(raw_val, str):
                    id_val = key_str
                    label = raw_val.strip()
                else:
                    continue

                if not label:
                    continue
                token_key = " ".join(_TOKEN_RE.findall(label.lower()))
                if not token_key:
                    continue
                entry = {
                    "pk_col": pk_col,
                    "id": id_val,
                    "label": label,
                }
                index.setdefault(token_key, []).append(entry)
    for token_key, entries in index.items():
        seen: set = set()
        deduped: List[dict] = []
        for e in entries:
            key = (e["pk_col"], e["id"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        index[token_key] = deduped

    log.info(
        "[reverse_lookup] Index built: %d unique labels, %d total entries",
        len(index),
        sum(len(v) for v in index.values()),
    )
    _vprint(
        "STARTUP", "blue",
        f" Reverse-lookup index: {len(index)} labels loaded",
        f"path={cache_path.name}",
    )
    return index

def reload_reverse_lookup_index() -> None:
    """Hot-reload the reverse-lookup index (e.g. after cache rebuild)."""
    global _REVERSE_LOOKUP_INDEX
    new_index = _build_reverse_lookup_index()
    with _reverse_lookup_lock:
        _REVERSE_LOOKUP_INDEX = new_index
    log.info("[reverse_lookup] Index reloaded: %d labels", len(_REVERSE_LOOKUP_INDEX))

def _reverse_lookup_question_values(
    question: str,
    relevant_tables: Optional[List[str]] = None,
) -> List[dict]:
    """
    Scan *question* for multi-word phrases that match known display labels.
    Returns list of: {"phrase", "pk_col", "id", "label"}
    Longest-match first; unigrams skipped.

    Ambiguity rule (checked in order):
    - Same label maps to DIFFERENT pk_col columns, and *relevant_tables* is
      known → keep only columns that actually exist on one of those tables
      (per FULL_SCHEMA_CACHE). A label like "Finished Goods" can legitimately
      appear on ITEM_TYPE_VSP_ID, ITEM_GROUP_VSP_ID, REPORT_STATUS_VSP_ID, etc.
      across unrelated tables — column-name heuristics alone can't tell those
      apart, but schema membership for the table actually being queried can.
    - Otherwise, attempt priority resolution: drop version/surrogate columns
      (_VERSION_ID, _ATTRIBUTE_ID) when a direct FK/PK column also matches.
      If a single winner remains, use it. If still multiple columns, skip.
    - Same label maps to same pk_col with multiple IDs → pick by exact label match.
    """
    with _reverse_lookup_lock:
        idx = _REVERSE_LOOKUP_INDEX
    if not idx:
        log.warning(
            "[reverse_lookup] Index is EMPTY (0 labels loaded) — "
            "reverse-lookup cannot run for question=%r. "
            "Check master_translation_cache.json path / reload endpoint.",
            question,
        )
        return []
    log.info(
        "[reverse_lookup] Scanning question=%r relevant_tables=%s index_size=%d labels",
        question, relevant_tables, len(idx),
    )
    _LOW_PRIORITY_SUFFIXES: tuple = ("_VERSION_ID", "_ATTRIBUTE_ID", "_MBOM_VERSION_ID")

    _table_cols: Optional[set] = None
    if relevant_tables:
        _table_cols = set()
        for _t in relevant_tables:
            _table_cols.update(
                c.upper() for c in FULL_SCHEMA_CACHE.get(_t, {}).get("columns", [])
            )

    tokens = re.findall(r"[a-zA-Z0-9''\-]+", question)
    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)
    matches: List[dict] = []
    consumed: set = set()
    for width in range(min(12, n), 1, -1):
        for start in range(n - width + 1):
            end = start + width
            if any(i in consumed for i in range(start, end)):
                continue
            phrase = " ".join(tokens_lower[start:end])
            if phrase not in idx:
                continue
            entries = idx[phrase]
            unique_cols = {e["pk_col"] for e in entries}

            if len(unique_cols) > 1:
                candidate_entries = entries
                candidate_cols = unique_cols

                # 1) Prefer columns that actually belong to the table(s) in scope.
                if _table_cols:
                    scoped = [
                        e for e in candidate_entries
                        if e["pk_col"].upper() in _table_cols
                    ]
                    scoped_cols = {e["pk_col"] for e in scoped}
                    if scoped and len(scoped_cols) == 1:
                        log.info(
                            "[reverse_lookup] Resolved ambiguity for %r via table scope %s → %s",
                            phrase, relevant_tables, next(iter(scoped_cols)),
                        )
                        candidate_entries = scoped
                        candidate_cols = scoped_cols
                    elif scoped:
                       
                        candidate_entries = scoped
                        candidate_cols = scoped_cols

                # 2) Priority-suffix filter (only needed if still ambiguous).
                if len(candidate_cols) > 1:
                    high = [
                        e for e in candidate_entries
                        if not any(e["pk_col"].upper().endswith(s) for s in _LOW_PRIORITY_SUFFIXES)
                    ]
                    if high:
                        candidate_entries = high
                        candidate_cols = {e["pk_col"] for e in candidate_entries}

                if len(candidate_cols) > 1:
                    log.info(
                        "[reverse_lookup] Ambiguous label %r — %d columns after priority filter %s, skipping",
                        phrase, len(candidate_cols), candidate_cols,
                    )
                    continue

                entries = candidate_entries
                log.info(
                    "[reverse_lookup] Resolved ambiguity for %r via column priority → %s",
                    phrase, next(iter(candidate_cols)),
                )
            _TOKEN_RE_INNER = re.compile(r"[a-zA-Z0-9''\-]+")
            exact: Optional[dict] = None
            for e in entries:
                token_label = " ".join(_TOKEN_RE_INNER.findall(e["label"].lower()))
                if token_label == phrase:
                    exact = e
                    break

            if exact is None:
                if len(entries) == 1:
                    exact = entries[0]
                else:
                    log.info(
                        "[reverse_lookup] Label %r — %d IDs for col %r, no exact match, skipping",
                        phrase, len(entries), entries[0]["pk_col"],
                    )
                    continue

            matches.append({
                "phrase": " ".join(tokens[start:end]),
                "pk_col": exact["pk_col"],
                "id": exact["id"],
                "label": exact["label"],
            })
            for i in range(start, end):
                consumed.add(i)
            log.info("[reverse_lookup] Matched %r → %s = %s", phrase, exact["pk_col"], exact["id"])
            _vprint("REVERSE", "cyan",
                    f" Reverse-lookup: '{exact['label']}' → {exact['pk_col']} = {exact['id']}")

    if not matches:
        log.info(
            "[reverse_lookup] No phrase in question=%r matched any known label "
            "(index_size=%d, relevant_tables=%s) — no reverse-lookup hint injected.",
            question, len(idx), relevant_tables,
        )
    return matches


_ENUM_FIX_TOKEN_RE = re.compile(r"[a-zA-Z0-9''\-]+")


def _get_column_entries(pk_col: str) -> List[dict]:
    """
    All {pk_col, id, label} entries in the reverse-lookup index for one column
    (case-insensitive), deduped by (id, label). Used once cross-column
    ambiguity is no longer a concern — i.e. the LLM has already committed to
    a specific column in the generated SQL, so we only need to check that
    column's own labels against the question.
    """
    pk_col_upper = pk_col.upper()
    with _reverse_lookup_lock:
        idx = _REVERSE_LOOKUP_INDEX
    out: List[dict] = []
    seen: set = set()
    for entries in idx.values():
        for e in entries:
            if e["pk_col"].upper() == pk_col_upper:
                key = (e["id"], e["label"].lower())
                if key not in seen:
                    seen.add(key)
                    out.append(e)
    return out


def _find_column_scoped_label_match(question: str, pk_col: str) -> Optional[dict]:
    """
    Find the longest phrase in *question* that matches a known display label
    for *pk_col*, with no cross-column ambiguity to resolve (the column is
    already fixed — it's whatever the LLM put in the generated SQL).

    Guards against a genuine data-quality problem in
    master_translation_cache.json: the same label text mapped to two
    different IDs under the SAME column (e.g. "Finished Goods" → 628 AND
    2841 under VALUESET_ID/REPORT_STATUS_VSP_ID). There's no way to know
    which is correct from the question alone, so that case is logged and
    skipped rather than guessed.
    """
    entries = _get_column_entries(pk_col)
    if not entries:
        return None

    by_label: Dict[str, List[dict]] = {}
    for e in entries:
        token_label = " ".join(_ENUM_FIX_TOKEN_RE.findall(e["label"].lower()))
        if token_label:
            by_label.setdefault(token_label, []).append(e)

    tokens = re.findall(r"[a-zA-Z0-9''\-]+", question)
    tokens_lower = [t.lower() for t in tokens]
    n = len(tokens_lower)

    for width in range(min(12, n), 1, -1):
        for start in range(n - width + 1):
            phrase = " ".join(tokens_lower[start:start + width])
            cands = by_label.get(phrase)
            if not cands:
                continue
            if len(cands) > 1:
                log.warning(
                    "[reverse_lookup][column_fix] Ambiguous label %r within column %r — "
                    "%d different IDs (%s) for identical label text. This is a "
                    "master_translation_cache.json data-quality issue (duplicate rows "
                    "sharing the same display text under one column) — skipping "
                    "auto-fix; dedupe the source valueset for this column.",
                    phrase, pk_col, len(cands), [c["id"] for c in cands],
                )
                return None
            return cands[0]
    return None


def _fix_enum_literals_by_question(sql: str, question: str) -> str:
    """
    Safety net for enum/FK/LOV columns that STAGE-0 reverse-lookup missed
    entirely — e.g. a label like "Finished Goods" was ambiguous across
    several tables/columns at that point in the pipeline, so no hint got
    injected and the LLM had to invent a numeric literal on its own
    (this is how `ITEM_TYPE_VSP_ID = 1` happened even though `1` isn't a
    valid id for that column anywhere in the cache).

    Runs unconditionally (not gated on _reverse_hits) because this is
    exactly the case where Stage 0 came back empty. Once the LLM has picked
    a column, cross-column ambiguity is gone — re-check the question against
    that column's own known labels and correct the literal if it's wrong or
    unrecognised.
    """
    if not sql:
        return sql
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
    except Exception as exc:
        log.warning("[fix_enum_literals] sqlglot parse failed (%s) — returning sql unchanged", exc)
        return sql

    changed = False

    def _col_name(node) -> Optional[str]:
        if isinstance(node, sqlglot.exp.Column):
            return node.this.this if hasattr(node.this, "this") else str(node.this)
        return None

    def _make_literal(id_val: str):
        return (
            sqlglot.exp.Literal.number(id_val)
            if str(id_val).lstrip("-").isdigit()
            else sqlglot.exp.Literal.string(id_val)
        )

    for eq in expr.find_all(sqlglot.exp.EQ):
        left, right = eq.left, eq.right
        col = _col_name(left)
        if col is None or not isinstance(right, sqlglot.exp.Literal) or not right.is_number:
            continue
        match = _find_column_scoped_label_match(question, col)
        if match is None:
            continue
        if str(right.this) != str(match["id"]):
            log.info(
                "[fix_enum_literals] %s = %s doesn't match cache id for the label named "
                "in the question → correcting to %s (%r)",
                col, right.this, match["id"], match["label"],
            )
            eq.set("right", _make_literal(match["id"]))
            changed = True

    for in_expr in expr.find_all(sqlglot.exp.In):
        col = _col_name(in_expr.this)
        if col is None:
            continue
        match = _find_column_scoped_label_match(question, col)
        if match is None:
            continue
        new_exprs = []
        in_changed = False
        for item in in_expr.expressions:
            if (
                isinstance(item, sqlglot.exp.Literal)
                and item.is_number
                and str(item.this) != str(match["id"])
            ):
                log.info(
                    "[fix_enum_literals] IN-list %s = %s doesn't match cache id for the "
                    "label named in the question → correcting to %s (%r)",
                    col, item.this, match["id"], match["label"],
                )
                new_exprs.append(_make_literal(match["id"]))
                in_changed = True
                changed = True
            else:
                new_exprs.append(item)
        if in_changed:
            in_expr.set("expressions", new_exprs)

    if not changed:
        return sql

    try:
        fixed = expr.sql(dialect="tsql")
        _vprint("REVERSE", "green", " Enum-literal cross-check fix applied (question vs cache)")
        log.info("[fix_enum_literals] Fixed SQL: %s", fixed[:200])
        return fixed
    except Exception as exc:
        log.warning("[fix_enum_literals] SQL serialisation failed (%s) — returning original", exc)
        return sql


# ── Embedding store — thread-safe singleton ───────────────────────────────────
class _RWLock:
    def __init__(self):
        self._read_ready = _threading.Condition(_threading.Lock())
        self._readers = 0
    def acquire_read(self):
        with self._read_ready:
            self._readers += 1
    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    def release_write(self):
        self._read_ready.release()
    class _ReadCtx:
        def __init__(self, lock): self._l = lock
        def __enter__(self): self._l.acquire_read(); return self
        def __exit__(self, *_): self._l.release_read()
    class _WriteCtx:
        def __init__(self, lock): self._l = lock
        def __enter__(self): self._l.acquire_write(); return self
        def __exit__(self, *_): self._l.release_write()
    def reading(self): return self._ReadCtx(self)
    def writing(self): return self._WriteCtx(self)

class EmbeddingStore:
    _instance: "EmbeddingStore | None" = None
    _class_lock = _threading.Lock()
    def __init__(self) -> None:
        self._rwlock = _RWLock()
        self._lov_label_embeddings: Dict[str, np.ndarray] = {}
        self._lov_label_list: List[str] = []
        self._lov_embed_matrix: Optional[np.ndarray] = None
        self._lov_context_cache: Dict[str, str] = {}
        self._table_doc_embeddings: Dict[str, List[float]]= {}
        self._phrase_to_label: Dict[str, str] = {}

    @classmethod
    def instance(cls) -> "EmbeddingStore":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def lov_embed_matrix(self) -> Optional[np.ndarray]:
        with self._rwlock.reading():
            return self._lov_embed_matrix

    @property
    def lov_label_list(self) -> List[str]:
        with self._rwlock.reading():
            return self._lov_label_list

    def get_lov_context(self, table: str) -> str:
        with self._rwlock.reading():
            return self._lov_context_cache.get(table, "")

    def get_phrase_to_label(self) -> Dict[str, str]:
        with self._rwlock.reading():
            return dict(self._phrase_to_label)

    def build_phrase_to_label(self, lov_data: dict) -> None:
        all_labels: set = set()
        for tlov in lov_data.values():
            for cmap in tlov.values():
                for lbl in cmap:
                    all_labels.add(lbl.lower())
        pm: Dict[str, str] = {}
        for lbl in all_labels:
            pm[lbl] = lbl
            pm[lbl.replace("-", " ")] = lbl
            pm[lbl.replace("-", "")] = lbl
            for w in lbl.replace("-", " ").split():
                if len(w) > 3:
                    pm[w] = lbl
            if lbl.endswith("ed"):
                pm[lbl[:-1]] = lbl
            if lbl.endswith("ing"):
                pm[lbl[:-3] + "e"] = lbl
        with self._rwlock.writing():
            self._phrase_to_label = pm

    def build_lov_label_embeddings(self, embedder: "SentenceTransformer", lov_data: dict) -> None:
        labels: set = set()
        for tlov in lov_data.values():
            for cmap in tlov.values():
                for lbl in cmap:
                    labels.add(lbl.lower())
        if not labels:
            return
        label_list = sorted(labels)
        vecs = embedder.encode(
            label_list,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        emb_map = {lbl: vec for lbl, vec in zip(label_list, vecs)}
        with self._rwlock.writing():
            self._lov_label_list = label_list
            self._lov_embed_matrix = vecs
            self._lov_label_embeddings = emb_map
        log.info(f"[EmbeddingStore] LOV embeddings: {len(label_list)} labels.")

    def build_table_embeddings(self, embedder: "SentenceTransformer", meta: dict) -> None:
        result: Dict[str, List[float]] = {}
        docs = {t: _build_table_document(m) for t, m in meta.items()}
        texts = [d for d in docs.values() if d.strip()]
        tables = [t for t, d in docs.items() if d.strip()]
        if texts:
            vecs = embedder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            for tbl, vec in zip(tables, vecs):
                result[tbl] = vec.tolist()
        with self._rwlock.writing():
            self._table_doc_embeddings = result
        log.info(f"[EmbeddingStore] Table embeddings: {len(result)} tables.")

    def build_lov_context_cache(self, lov_data: dict, meta: dict, system_status_cols: set) -> None:
        cache: Dict[str, str] = {}
        for table in meta.keys():
            lov = lov_data.get(table, {})
            lines: List[str] = []
            if lov:
                lines.append(
                    f"LOV VALUES for {table} — use numeric IDs ONLY. "
                    "NEVER use string abbreviations like 'A' or 'I'; always use the integer ID shown below:"
                )
                for col, mapping in lov.items():
                    if not mapping:
                        continue
                    pairs = [_lov_label_value(rk, m) for rk, m in mapping.items()]
                    lines.append(
                        " " + col + ": "
                        + " | ".join(f"{l} → {v}" for l, v in pairs)
                    )
            type_ctx = get_type_lov_context(table)
            if type_ctx:
                if lines:
                    lines.append("")
                lines.append(type_ctx)
            if lines:
                cache[table] = "\n".join(lines)
        with self._rwlock.writing():
            self._lov_context_cache = cache
        log.info(f"[EmbeddingStore] LOV context cache: {len(cache)} tables.")

    def fuzzy_lov_match(self, phrase: str, embedder, threshold: float = _LOV_FUZZY_THRESHOLD) -> Optional[str]:
        with self._rwlock.reading():
            mat = self._lov_embed_matrix
            lbl_list = self._lov_label_list
        if mat is None or embedder is None:
            return None
        q_vec = embedder.encode(phrase, normalize_embeddings=True)
        sims = mat @ q_vec
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= threshold:
            return lbl_list[best_idx]
        return None

    def batch_fuzzy_lov_match(
        self, phrases: List[str], embedder, threshold: float = _LOV_FUZZY_THRESHOLD
    ) -> Dict[str, Optional[str]]:
        with self._rwlock.reading():
            mat = self._lov_embed_matrix
            lbl_list = self._lov_label_list
        if mat is None or embedder is None or not phrases:
            return {p: None for p in phrases}
        vecs = embedder.encode(phrases, normalize_embeddings=True)
        sims = vecs @ mat.T
        results: Dict[str, Optional[str]] = {}
        for i, phrase in enumerate(phrases):
            best_idx = int(np.argmax(sims[i]))
            results[phrase] = lbl_list[best_idx] if float(sims[i][best_idx]) >= threshold else None
        return results


# ── LOV data ─────────────────────────────────────────────────────────────────
def load_lov_data():
    try:
        return json.loads((_DATA / "lov_mapping_dynamic.json").read_text(encoding="utf-8"))
    except Exception:
        log.warning("lov_mapping_dynamic.json not found — LOV resolution disabled.")
        return {}

LOV_DATA: dict = load_lov_data()
_lov_data_lock = _threading.Lock()

def reload_lov_data():
    global LOV_DATA
    new_data = load_lov_data()
    with _lov_data_lock:
        LOV_DATA = new_data
    _s = EmbeddingStore.instance()
    _s.build_phrase_to_label(LOV_DATA)
    if _global_embedder is not None:
        _s.build_lov_label_embeddings(_global_embedder, LOV_DATA)
    _s.build_lov_context_cache(LOV_DATA, META, _SYSTEM_STATUS_COLS)
    _translator_load_lov_map()
    log.info("[admin] LOV data and caches reloaded.")

def _build_lov_label_embeddings(embedder: SentenceTransformer) -> None:
    EmbeddingStore.instance().build_lov_label_embeddings(embedder, LOV_DATA)
    _vprint("STARTUP", "blue",
            f" LOV fuzzy embeddings ready — "
            f"{len(EmbeddingStore.instance().lov_label_list)} labels")

def _fuzzy_lov_match(phrase: str, threshold: float = _LOV_FUZZY_THRESHOLD) -> Optional[str]:
    return EmbeddingStore.instance().fuzzy_lov_match(phrase, _global_embedder, threshold)

def _canonical_label(text: str) -> Optional[str]:
    return EmbeddingStore.instance().get_phrase_to_label().get(text.lower())

def get_lov_for_table(table: str) -> dict:
    return LOV_DATA.get(table, {})

def _lov_label_value(raw_key, meta) -> Tuple[str, str]:
    """
    lov_mapping_dynamic.json stores each column mapping as:
        { <raw_stored_value>: {"param_id": ..., "valueset_id": ...,
                                "valueset_name": ..., "description": <human label>} }
    i.e. keyed by the *raw stored value*, with the human-readable label inside
    meta["description"]. Returns (human_label, actual_stored_value) so callers
    can index/match on the label while still writing the real stored value into SQL.
    Falls back to treating (raw_key, meta) as a flat (label, value) pair if meta
    isn't a dict, so older/flat-format LOV files keep working.
    """
    if isinstance(meta, dict):
        label = meta.get("description") or str(raw_key)
        value = raw_key
    else:
        label = raw_key
        value = meta
    return str(label), str(value)


# ── System status cols ────────────────────────────────────────────────────────
def _derive_system_status_cols() -> set:
    if not META:
        return {"status"}
    total_tables = len(META)
    col_table_count: Dict[str, int] = {}
    col_dtypes: Dict[str, set] = {}
    _flag_suffixes = (
        "_flag", "_active", "_enabled", "_disabled", "_status",
        "_toggle", "_switch", "_ind", "_indicator", "_yn",
        "_bool", "_bit", "_on", "_off",
    )
    _flag_dtypes = {
        "bit", "tinyint", "smallint", "boolean", "bool",
        "int", "integer", "number", "numeric",
    }
    for _tbl, _tinfo in META.items():
        for _col, _cinfo in _tinfo.get("columns", {}).items():
            _cl = _col.lower()
            col_table_count[_cl] = col_table_count.get(_cl, 0) + 1
            _dtype = (
                (_cinfo.get("data_type") or "") if isinstance(_cinfo, dict) else ""
            ).lower()
            col_dtypes.setdefault(_cl, set()).add(_dtype)
    result: set = set()
    for _col, _cnt in col_table_count.items():
        if len(_col) <= 10 and _cnt / max(total_tables, 1) >= 0.40:
            result.add(_col)
            continue
        if any(_col.endswith(_sfx) for _sfx in _flag_suffixes):
            if col_dtypes.get(_col, set()) & _flag_dtypes:
                result.add(_col)
    if not result:
        fallback = sorted(
            [(c, n) for c, n in col_table_count.items() if len(c) <= 12],
            key=lambda x: x[1], reverse=True,
        )
        if fallback:
            result.add(fallback[0][0])
    return result

_SYSTEM_STATUS_COLS: set = _derive_system_status_cols()

def _get_system_status_labels() -> set:
    labels: set = set()
    for _tlov in LOV_DATA.values():
        for col, mapping in _tlov.items():
            if col.lower() in _SYSTEM_STATUS_COLS:
                for lbl in mapping:
                    ll = lbl.lower()
                    labels.add(ll)
                    labels.add(ll.replace("-", " "))
                    labels.add(ll.replace(" ", "-"))
                    labels.add(ll.replace("-", ""))
    return labels

def _question_implies_system_status(question: str) -> bool:
    triggers = _get_system_status_labels()
    if not triggers:
        return False
    q = question.lower()
    return any(ph in q for ph in triggers)


# ── Join graph ─────
from core.graph_join_resolver import load_all_join_graphs as _load_all_join_graphs
try:
    _JD: dict = _load_all_join_graphs()
    log.info(
        "Join graph loaded: %d base tables, %d graph nodes.",
        len(_JD.get("join_paths", {})),
        len(_JD.get("graph", {})),
    )
except Exception as _jg_exc:
    _JD = {}
    log.warning("Join graph load failed (%s) — joins unavailable.", _jg_exc)

_graph_resolver: Optional["GraphJoinResolver"] = None

# ══════════════════════════════════════════════════════════════════════════
# LLM-BASED TABLE SELECTION
# ─────────────────────────────────────────────────────────────────────────


_TABLE_SELECTOR_INTENT_LABELS: List[str] = [
    "simple", "lookup", "aggregation", "group_by", "sort_by",
    "existence", "conditional", "time_based", "top_bottom", "search",
    "relational_lookup", "relational_aggregation", "relational",
]

_TABLE_SELECTOR_CONFIDENCE_LABELS: List[str] = ["high", "low"]


def _get_subgroup_tables(module_name: str, subgroup_name: str) -> Optional[List[dict]]:
    """
    Look up the raw table-metadata dicts for one module/subgroup out of
    MODULE_DATA (mirrors _build_narrow_table_index's traversal). Returns
    None if the module or subgroup key doesn't exist.
    """
    mod_val = MODULE_DATA.get("modules", {}).get(module_name)
    if not mod_val:
        return None
    sg_val = mod_val.get("subgroups", {}).get(subgroup_name)
    if sg_val is None:
        return None
    sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
    return [t for t in sg_list if isinstance(t, dict) and t.get("table_name")]


def _format_table_catalog_for_llm(tables: List[dict]) -> str:
    """
    Renders this subgroup's tables into a compact, LLM-readable block using
    the real narrow-json fields (business_purpose / row_represents /
    synonyms / use_when / do_not_use_when / parent_entity / columns),
    the same fields _build_table_document() uses for the embedding index.
    """
    lines: List[str] = []
    for tbl in tables:
        name = tbl.get("table_name", "")
        if not name:
            continue

        bits: List[str] = []

        for field in ("business_purpose", "row_represents"):
            val = tbl.get(field, "")
            if val:
                bits.append(str(val))

        synonyms = tbl.get("synonyms") or []
        if synonyms:
            bits.append("aka: " + ", ".join(synonyms))

        use_when = tbl.get("use_when") or []
        if use_when:
            bits.append("use when: " + "; ".join(use_when))

        do_not_use = tbl.get("do_not_use_when") or []
        if do_not_use:
            bits.append("not for: " + "; ".join(do_not_use))

        parent = tbl.get("parent_entity")
        if parent:
            bits.append(f"parent: {parent}")

        cols = tbl.get("columns")
        if isinstance(cols, dict):
            col_names = list(cols.keys())
        elif isinstance(cols, list):
            col_names = list(cols)
        else:
            col_names = []
        if col_names:
            preview = ", ".join(col_names[:_TABLE_SELECTOR_MAX_COLS_HINT])
            more = (
                f" (+{len(col_names) - _TABLE_SELECTOR_MAX_COLS_HINT} more)"
                if len(col_names) > _TABLE_SELECTOR_MAX_COLS_HINT else ""
            )
            bits.append(f"cols: {preview}{more}")

        line = f"- {name}" + (f": {' | '.join(bits)}" if bits else "")
        lines.append(line)

    text = "\n".join(lines)
    if len(text) > _TABLE_SELECTOR_MAX_CATALOG_CHARS:
        text = text[:_TABLE_SELECTOR_MAX_CATALOG_CHARS] + "\n... (truncated)"
    return text


def _format_join_edges_for_llm(table_names: List[str]) -> str:
    """
    Renders join relationships between `table_names` using the already-loaded
    merged join graph _JD["join_paths"], restricted to edges where both ends
    are in this subgroup (keeps the prompt small — a subgroup's table set is
    usually a fraction of the full cross-module join graph).
    """
    name_set = {t.upper() for t in table_names}
    join_paths = _JD.get("join_paths", {}) if isinstance(_JD, dict) else {}
    lines: List[str] = []
    seen: set = set()

    for base_tbl, targets in join_paths.items():
        if base_tbl.upper() not in name_set or not isinstance(targets, dict):
            continue
        for tgt_tbl, conditions in targets.items():
            if tgt_tbl.upper() not in name_set or not isinstance(conditions, list):
                continue
            for cond in conditions:
                ft = cond.get("from_table", base_tbl)
                fc = cond.get("from_column", "?")
                tt = cond.get("to_table", tgt_tbl)
                tc = cond.get("to_column", "?")
                key = tuple(sorted((f"{ft}.{fc}", f"{tt}.{tc}")))
                if key not in seen:
                    seen.add(key)
                    lines.append(f"{ft}.{fc} = {tt}.{tc}")

    if not lines:
        return "(no known join relationships among this subgroup's tables)"
    text = "\n".join(lines)
    if len(text) > _TABLE_SELECTOR_MAX_CATALOG_CHARS:
        text = text[:_TABLE_SELECTOR_MAX_CATALOG_CHARS] + "\n... (truncated)"
    return text


_SUBMODULE_FILTER_STOPWORDS = {
    "question", "about", "table", "tables", "module", "questions", "which",
    "what", "when", "where", "with", "from", "this", "that", "into", "over",
    "have", "been", "were", "than", "then", "only", "also", "used", "using",
    "need", "needs", "needed", "current", "value", "values",
}


def _tokenize_for_overlap(text: str) -> Set[str]:
    return {
        w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if w not in _SUBMODULE_FILTER_STOPWORDS
    }


def _score_submodules_by_keyword_overlap(
    question: str, submodule_names: List[str],
) -> List[str]:
    """
    Cheap lexical prefilter run BEFORE the LLM submodule classifier.

    Why this exists: with the full submodule list handed to a small local
    model, it tends to default to the first option whenever it's unsure —
    rather than correctly returning "0" (no clear match) as instructed.
    For a container like PROD, submodule #1 happens to be "items" (whose
    entire concept vocabulary is bridge-only MFC_* item-master tables),
    so an unsure classifier was silently, systematically routing questions
    to it and returning nothing but MFC tables regardless of what was
    actually asked.

    This scores each submodule's own concept text by raw word overlap with
    the question and drops anything with zero overlap, so a submodule that
    shares no vocabulary with the question (e.g. "items" for a "rework
    order serial number" question) is never even offered to the LLM as an
    option — removing the failure mode at its root instead of hoping the
    model picks correctly among candidates that were never relevant.

    Returns submodule_names filtered to score > 0, sorted by score
    descending, capped to keep the classification prompt small. Empty
    result means genuinely no lexical match anywhere — callers should skip
    LLM classification and fall back to the container-level flow.
    """
    q_tokens = _tokenize_for_overlap(question)
    if not q_tokens:
        return submodule_names[:8]

    scored: List[Tuple[str, int]] = []
    for name in submodule_names:
        rule_text = MODULE_PROMPT_RULES.get(name, "")
        sub_tokens = _tokenize_for_overlap(rule_text)
        score = len(q_tokens & sub_tokens)
        if score > 0:
            scored.append((name, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [name for name, _score in scored[:8]]


def _build_submodule_classification_prompt(
    question: str, module_name: str, submodule_names: List[str],
) -> str:
    listing = "\n".join(f"{i}. {name}" for i, name in enumerate(submodule_names, start=1))
    return f"""You are a routing assistant for a natural-language-to-SQL system.

Module: {module_name}

This module is split into the following submodules, numbered — this is the
COMPLETE list. Do not invent, rename, merge, or guess a submodule that is
not in this list:
{listing}

User question:
"{question}"

Task:
Pick the ONE submodule NUMBER above that this question is primarily about.
- Only ever pick a number from the list given above.
- If the question doesn't clearly belong to exactly one submodule (it spans
  several, or none of them fit), return 0.

Respond with ONLY a JSON object, no explanation, no markdown fences, in exactly this shape:
{{"submodule": 2}}
"""


async def _classify_submodule_llm(
    question: str, module_name: str, submodule_names: List[str],
) -> Optional[str]:
    """
    Pre-narrowing step that runs BEFORE Call A (concept extraction) when a
    question has no subgroup and its module is a multi-submodule container
    (e.g. "PROD" bundling rework_order, wo_item_serialization, items, ...).

    Without this, Call A's concept vocabulary is the WHOLE container's
    concept list concatenated across every submodule — 90+ numbered
    near-duplicate bullets for a container like PROD — which a small local
    model (currently XiYanSQL-7B, formerly qwen2.5:3b) discriminates against
    poorly. Routing to a single
    submodule's own (much smaller, ~5-15 concept) vocabulary first makes
    Call A's job tractable again.

    Same "pick by NUMBER" pattern as _extract_concepts_llm, for the same
    reason: far more reliable for a small LLM than reproducing a submodule
    name verbatim, and "invented submodule" reduces to a trivial range
    check.

    Returns the chosen submodule key (already validated to be one of
    submodule_names), or None if the model returned 0, an invalid/
    out-of-range number, or the call failed outright — callers must treat
    None as "could not confidently classify" and fall back to the existing
    container-level concept flow, not as an error.
    """
    if len(submodule_names) < 2:
        return None

    filtered_names = _score_submodules_by_keyword_overlap(question, submodule_names)
    if not filtered_names:
        _vprint("TABLESEL", "cyan",
                " Submodule classifier: no keyword overlap with any submodule",
                f"module={module_name} — using container-level concept vocabulary")
        return None
    if len(filtered_names) == 1:

        _vprint("TABLESEL", "cyan",
                f" Submodule classifier: unambiguous keyword match → {filtered_names[0]!r}",
                f"module={module_name} (skipped LLM call)")
        return filtered_names[0]

    prompt = _build_submodule_classification_prompt(question, module_name, filtered_names)
    parsed = await _call_table_selector_llm(prompt)
    if not parsed:
        log.warning(
            "[submodule_classifier_llm] No usable output for module=%r — "
            "falling back to container-level concept vocabulary",
            module_name,
        )
        return None

    raw_pick = parsed.get("submodule")
    try:
        idx = int(raw_pick)
    except (TypeError, ValueError):
        log.warning(
            "[submodule_classifier_llm] Model returned non-numeric submodule %r for "
            "module %r — falling back to container-level concept vocabulary",
            raw_pick, module_name,
        )
        return None

    if idx == 0:
        _vprint("TABLESEL", "cyan",
                " Submodule classifier: question spans/doesn't fit a single submodule",
                f"module={module_name} — using container-level concept vocabulary")
        return None

    if not (1 <= idx <= len(filtered_names)):
        log.warning(
            "[submodule_classifier_llm] Model returned out-of-range submodule number %r "
            "(module %r had %d filtered candidate(s)) — falling back to container-level concept vocabulary",
            idx, module_name, len(filtered_names),
        )
        return None

    chosen = filtered_names[idx - 1]
    _vprint("TABLESEL", "cyan",
            f" Submodule classifier picked: {chosen!r}",
            f"module={module_name} (narrowed from {len(filtered_names)} keyword-filtered candidate(s), {len(submodule_names)} total)")
    return chosen


def _build_concept_extraction_prompt(
    question: str, module_name: str, concept_rules: List[Tuple[str, List[str]]],
) -> str:
    vocab_text = "\n".join(f"{i}. {desc}" for i, (desc, _tables) in enumerate(concept_rules, start=1))
    return f"""You are a business-concept extraction assistant for a natural-language-to-SQL system.

Module: {module_name}

Predefined business concepts for this module, numbered — this is the
COMPLETE list. Do not invent, rename, merge, or guess a concept that is
not in this list:
{vocab_text}

User question:
"{question}"

Task:
Pick every concept NUMBER above that this question is actually asking about.
- Only ever pick numbers from the list given above.
- Pick the smallest set of numbers that covers what the question needs.
- If none of the listed concepts apply to this question, return an empty list.

Respond with ONLY a JSON object, no explanation, no markdown fences, in exactly this shape:
{{"concepts": [1, 3]}}
"""


async def _extract_concepts_llm(
    question: str, module_name: str, concept_rules: List[Tuple[str, List[str]]],
) -> Tuple[List[int], bool]:
    """
    Call A of the table-selection flow: reuses the exact same LLM/model as
    the final table selector (Call B) — same _call_table_selector_llm —
    just with a narrower prompt asking "which numbered concept(s) from
    this module's own rule text does this question touch?".

    Concepts are selected by NUMBER rather than by asking the model to
    reproduce concept text verbatim — far more reliable for an LLM to get
    right, and it also means "invented concept" reduces to a trivial
    range check instead of fuzzy string matching.

    Returns (indices, ok):
      - indices: validated 1-based positions into `concept_rules`. Any
        number outside range, or non-numeric junk, is dropped and logged
        — never trusted downstream.
      - ok: False only on a hard LLM failure (timeout / no response /
        unparseable JSON). The caller must treat ok=False as "extraction
        didn't run" and fall back to the existing full-subgroup flow — it
        is NOT the same as the model legitimately returning zero concepts.
    """
    prompt = _build_concept_extraction_prompt(question, module_name, concept_rules)
    parsed = await _call_table_selector_llm(prompt)
    if not parsed:
        log.warning(
            "[concept_extractor_llm] No usable output for module=%r — "
            "caller will fall back to full-subgroup table selection",
            module_name,
        )
        return [], False

    raw_concepts = parsed.get("concepts") or []
    n = len(concept_rules)
    indices: List[int] = []
    for c in raw_concepts:
        try:
            idx = int(c)
        except (TypeError, ValueError):
            log.warning(
                "[concept_extractor_llm] Model returned non-numeric concept %r for "
                "module %r — dropping",
                c, module_name,
            )
            continue
        if 1 <= idx <= n and idx not in indices:
            indices.append(idx)
        else:
            log.warning(
                "[concept_extractor_llm] Model returned out-of-range concept number %r "
                "(module %r has %d concept(s)) — dropping, not invented, not used",
                idx, module_name, n,
            )

    _vprint("TABLESEL", "cyan",
            f" Concept extraction (Call A): {[concept_rules[i-1][0] for i in indices]}",
            f"module={module_name} vocab_size={n}")
    return indices, True


def _expand_required_tables_shortest_path(
    required_tables: List[str], allowed_universe: List[str],
) -> Set[str]:
    """
    Controlled join-graph expansion for the concept-narrowed candidate set.

    This is deliberately NOT the same thing as get_neighbor_tables() /
    Tier-2 BFS in build_join_clause, which blast outward N hops from one
    base table and return every neighbor. Here we already know the full
    set of tables that MUST be in the final candidate list (one or more
    per matched concept) — the only open question is which intermediate
    tables are needed to connect them. So for every required table beyond
    the first, we BFS for the single shortest path back to the anchor,
    restricted to `allowed_universe` (this subgroup's own table set) so
    nothing outside the subgroup leaks into the LLM's candidate list, and
    add only the tables that sit ON that path — never a full neighbor
    expansion.

    Uses the same _JD["join_paths"] structure and _JOIN_MAX_BFS_DEPTH cap
    as the existing Tier 2 BFS in build_join_clause, just scoped to a
    restricted universe and a set of must-include tables instead of one
    base + arbitrary targets.
    """
    req = list(dict.fromkeys(required_tables)) 
    result: Set[str] = set(req)
    if len(req) < 2:
        return result

    universe = {t.upper() for t in allowed_universe}
    _JP = _JD.get("join_paths", {})
    anchor = req[0]

    for tgt in req[1:]:
        if tgt.upper() == anchor.upper():
            continue

        path = _JP.get(anchor, {}).get(tgt)
        if path is None:
            q: deque = deque([(anchor, [])])
            visited = {anchor.lower()}
            while q:
                cur, acc = q.popleft()
                if len(acc) >= _JOIN_MAX_BFS_DEPTH:
                    continue
                for tt, steps in _JP.get(cur, {}).items():
                    if universe and tt.upper() not in universe:
                        continue  
                    new_acc = acc + [steps]
                    if tt.lower() == tgt.lower():
                        path = [s for seg in new_acc for s in seg]
                        break
                    if tt.lower() not in visited:
                        visited.add(tt.lower())
                        q.append((tt, new_acc))
                if path:
                    break

        if path:
            for step in path:
                result.add(step["from_table"])
                result.add(step["to_table"])
        else:
            log.warning(
                "[concept_join_expand] No shortest path found from %r to %r within "
                "the allowed subgroup universe — leaving them unconnected; the final "
                "table selector / join builder will surface this downstream",
                anchor, tgt,
            )

    return result


def _build_table_selector_prompt(
    question: str, module_name: str, catalog_text: str, join_edges_text: str,
    module_rule_text: str = "", pre_narrowed: bool = False,
) -> str:
    rule_block = f"\nModule-specific table hierarchy rules (follow these strictly):\n{module_rule_text}\n" if module_rule_text else ""
 
    if pre_narrowed:
        smallest_set_line = (
            "   - This candidate list has ALREADY been filtered down to the "
            "tables judged necessary for this question — do not drop any of "
            "them without a clear, specific reason tied to the question text. "
            "Only exclude a listed table if it plainly cannot contribute to "
            "the answer."
        )
    else:
        smallest_set_line = "   - Prefer the smallest table set that fully answers the question."
    max_tables_line = (
        "   - Pick AT MOST 4 tables total, no matter how many are available. "
        "If the question genuinely cannot be answered within 4 tables, pick "
        "the 4 most essential ones rather than exceeding the limit."
    )
    return f"""You are a table-selection assistant for a natural-language-to-SQL system.

Module: {module_name}

Available tables in this module:
{catalog_text}

Known join relationships between tables in this module:
{join_edges_text}
{rule_block}
User question:
"{question}"

Task:
1. Decide whether this question is a legitimate manufacturing/MES data
   question that could plausibly be answered using tables from this module
   — set "in_scope" to true — or whether it's off-topic / not a data
   question at all (e.g. small talk, general knowledge, unrelated chit-chat)
   — set "in_scope" to false.
2. Decide the query intent — pick exactly one from this list:
   {", ".join(_TABLE_SELECTOR_INTENT_LABELS)}
3. Decide the final set of table(s) needed to answer the question.
   - Pick only tables that are actually needed.
   - If more than one table is needed, only include tables that have a
     known join relationship to at least one other selected table.
{smallest_set_line}
{max_tables_line}
   - If module-specific hierarchy rules are given above, they take priority
     over the general guidance below — follow them exactly, including which
     parent/root table must accompany a child table.
4. Rate your own confidence that the tables listed above (this module's
   catalog) are actually the right place to answer this question — not just
   your confidence in which specific tables you picked. Set "confidence" to
   "low" if the question doesn't fit well within this module's tables at all
   (it may belong to a different module/subgroup), otherwise "high".

Respond with ONLY a JSON object, no explanation, no markdown fences, in exactly this shape:
{{"in_scope": true, "intent": "<one of the intent labels>", "tables": ["TABLE_ONE", "TABLE_TWO"], "confidence": "<high or low>"}}
"""


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Scan `text` for the first balanced {...} object, respecting string
    literals (so a brace inside a quoted value doesn't throw off the
    count). Returns the substring, or None if no balanced object is found.

    This replaces a naive greedy regex (`\\{.*\\}`), which — because it's
    greedy — grabs from the *first* '{' to the *last* '}' in the whole
    string. That breaks the moment the model adds any trailing prose
    containing a stray '}', or emits more than one brace-y chunk.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_table_selector_output(raw: str) -> Optional[dict]:
    text = raw.strip()


    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_text = _extract_first_json_object(text)
    if obj_text:
        try:
            return json.loads(obj_text)
        except json.JSONDecodeError:
            pass

    return None


async def _call_table_selector_llm(prompt: str) -> Optional[dict]:
    """
    Single best-effort call to the table-selector model (currently
    XiYanSQL-7B; formerly qwen2.5:3b — same call site, model swapped in
    the same CPU slot) for table + intent selection. Returns the parsed
    {"intent": ..., "tables": [...]} dict, or None on any failure
    (non-200, timeout, unparseable JSON) so the caller can fail open.

    Backend-aware: the table-selector model runs on a llama.cpp server
    (CPU, port 8081, _LLM_SERVER_URL pointing at its /completion endpoint).
    The full-size XiYanSQL SQL-generation model is a SEPARATE instance
    served over the GPU service (_GPU_SERVICE_URL) — the two are unrelated
    processes even though they're both "XiYanSQL". LLM_BACKEND selects
    which request/response shape to use here (llamacpp vs Ollama).
    """
    if _LLM_BACKEND == "llamacpp":
        payload = {
            "prompt": prompt,
            "n_predict": _TABLE_SELECTOR_MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        payload = {
            "model": _TABLE_SELECTOR_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": _TABLE_SELECTOR_MAX_TOKENS,
                "temperature": 0.0,
            },
        }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _LLM_SERVER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_TABLE_SELECTOR_TIMEOUT_SECS),
            ) as resp:
                if resp.status != 200:
                    log.warning("[table_selector_llm] Server returned HTTP %d", resp.status)
                    return None
                data = await resp.json()
                raw_field = "content" if _LLM_BACKEND == "llamacpp" else "response"
                raw = (data.get(raw_field) or "").strip()
                if not raw:
                    log.warning("[table_selector_llm] Empty response from model")
                    return None
                parsed = _parse_table_selector_output(raw)
                if parsed is None:
                    log.warning("[table_selector_llm] Could not parse JSON from: %.200s", raw)
                return parsed
    except asyncio.TimeoutError:
        log.warning("[table_selector_llm] Timed out after %.0fs", _TABLE_SELECTOR_TIMEOUT_SECS)
        return None
    except Exception as exc:
        log.warning("[table_selector_llm] Call failed (non-fatal): %s", exc)
        return None


async def _call_chat_topic_llm(question: str) -> Optional[str]:
    """
    One best-effort call to the same small on-prem model used for table
    selection above, asked instead to turn a user's question into a short
    ChatGPT-style topic ("Show Item Types", not the question restated
    verbatim). Called once per conversation — see is_new_chat handling in
    _run_query_pipeline — never on every message.

    Purely cosmetic: on any failure (timeout, bad response, empty output)
    this returns None and the caller falls back to chat_history.py's plain
    first-80-chars title, so a slow/down LLM never blocks chat saving.
    """
    prompt = (
        "Turn the user question below into a short topic title for a chat "
        "history sidebar: 2 to 4 words, Title Case, no punctuation, no "
        "quotes. Reply with ONLY the title, nothing else.\n\n"
        f"Question: {question.strip()[:300]}\nTopic:"
    )
    if _LLM_BACKEND == "llamacpp":
        payload = {
            "prompt": prompt,
            "n_predict": _CHAT_TOPIC_MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        payload = {
            "model": _TABLE_SELECTOR_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": _CHAT_TOPIC_MAX_TOKENS,
                "temperature": 0.0,
            },
        }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _LLM_SERVER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_CHAT_TOPIC_TIMEOUT_SECS),
            ) as resp:
                if resp.status != 200:
                    log.warning("[chat_topic_llm] Server returned HTTP %d", resp.status)
                    return None
                data = await resp.json()
                raw_field = "content" if _LLM_BACKEND == "llamacpp" else "response"
                raw = (data.get(raw_field) or "").strip()
                if not raw:
                    log.warning("[chat_topic_llm] Empty response from model")
                    return None
                # Small models sometimes echo the "Topic:" label, wrap the
                # answer in quotes, or add a trailing line — clean defensively.
                topic = raw.splitlines()[0].split("Topic:")[-1].strip().strip('"\'.,').strip()
                return topic[:_CHAT_TOPIC_MAX_CHARS] if topic else None
    except asyncio.TimeoutError:
        log.warning("[chat_topic_llm] Timed out after %.0fs", _CHAT_TOPIC_TIMEOUT_SECS)
        return None
    except Exception as exc:
        log.warning("[chat_topic_llm] Call failed (non-fatal): %s", exc)
        return None


async def resolve_tables_llm(
    question: str,
    module_name: str,
    subgroup_name: str,
    flat_table_names: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[str], Optional[str], bool, str]:
    """
    LLM-based table + intent selector. The model is the sole authority on
    which tables answer the question — there is no scoring, boosting,
    pinning, or probability filtering downstream of this call, and there is
    no silent "return every table" fallback when the model's output is
    missing or unparseable. If the model can't produce a usable pick, that's
    surfaced as a real error (empty tables list + reason) so the endpoint
    can tell the user, rather than quietly guessing with the full table set.

    The model also returns an in_scope flag and a self-rated confidence
    signal as part of the same call (no extra network round-trip), but
    the caller no longer gates on either — QA/subgroup validation was
    removed; only `tables`, `intent`, and `error` are used downstream.

    Returns (tables, intent, error, in_scope, confidence).
    `error` is None on success; when it is set, `tables` is always [] and
    the caller should fail the request.

    Flow — a SINGLE LLM call, no concept-extraction pre-pass:
        Question -> Module -> Subgroup
          -> Full subgroup/module table catalog, guided by a fixed
             MFC/PRD/SEQ domain hint (_TABLE_SELECTOR_DOMAIN_HINT)
          -> One table-selector call picks the final table(s) + intent
          -> Final tables -> existing join builder -> SQL generation

    `flat_table_names`: for modules with no subgroup config (subgroup_name
    is empty), the caller passes the module's flat table list here instead.
    The candidate table-metadata dicts are then looked up directly out of
    META (populated from that module's *_narrow.json) rather than via
    _get_subgroup_tables(), so table selection still runs against that
    flat list instead of the caller being forced to skip selection and
    hand the pipeline every table unfiltered.
    """
    if subgroup_name:
        subgroup_tables = _get_subgroup_tables(module_name, subgroup_name)
        if subgroup_tables is None:
            return [], None, f"Subgroup '{subgroup_name}' not found under module '{module_name}'.", True, "high"
    else:
        names = flat_table_names or []
        subgroup_tables = [META[n] for n in names if n in META]
        if not subgroup_tables:
            return [], None, f"None of the provided tables for module '{module_name}' have metadata loaded (check data/{module_name.lower()}_narrow.json).", True, "high"

    all_table_names = [t["table_name"] for t in subgroup_tables]
    if not all_table_names:
        return [], None, f"Subgroup '{subgroup_name}' under module '{module_name}' has no tables.", True, "high"

    if not _TABLE_SELECTOR_ENABLED:
        return list(all_table_names), None, None, True, "high"

 
    candidate_tables = subgroup_tables


    _rule_lookup_key = subgroup_name or module_name
    module_rule_text = _get_table_selector_rule_text(_rule_lookup_key)

    async def _run_call_b(cb_candidate_tables: list) -> Tuple[List[str], Optional[str], bool, str, Optional[dict]]:
        """Runs the final table-selector LLM call (Call B) against whatever
        candidate table list is passed in, and validates its picks against
        that same list. Returns (picked, llm_intent, in_scope, confidence, parsed)."""
        cb_candidate_names = [t["table_name"] for t in cb_candidate_tables]
        cb_catalog_text = _format_table_catalog_for_llm(cb_candidate_tables)
        cb_join_edges_text = _format_join_edges_for_llm(cb_candidate_names)
        cb_prompt = _build_table_selector_prompt(
            question, module_name, cb_catalog_text, cb_join_edges_text, module_rule_text,
            pre_narrowed=False,
        )
        cb_parsed = await _call_table_selector_llm(cb_prompt)
        if not cb_parsed:
            return [], None, True, "high", None

        cb_valid_by_upper = {t.upper(): t for t in cb_candidate_names}
        cb_picked_raw = cb_parsed.get("tables") or []
        cb_picked: List[str] = []
        for t in cb_picked_raw:
            canonical = cb_valid_by_upper.get(str(t).upper())
            if canonical and canonical not in cb_picked:
                cb_picked.append(canonical)
            elif not canonical:
                log.warning(
                    "[table_selector_llm] Model picked table %r not in subgroup '%s/%s' — dropping",
                    t, module_name, subgroup_name,
                )

        cb_raw_in_scope = cb_parsed.get("in_scope")
        cb_in_scope: bool = True if cb_raw_in_scope is None else bool(cb_raw_in_scope)

        cb_raw_confidence = cb_parsed.get("confidence")
        cb_confidence: str = cb_raw_confidence if cb_raw_confidence in _TABLE_SELECTOR_CONFIDENCE_LABELS else "high"
        return cb_picked, cb_parsed.get("intent"), cb_in_scope, cb_confidence, cb_parsed

    picked, raw_intent, in_scope, confidence, parsed = await _run_call_b(candidate_tables)

    if parsed is None:
        log.warning("[table_selector_llm] No usable output from model — failing the request")
        return [], None, "Table selector could not parse a response for this question. Please rephrase.", True, "high"

    if not in_scope:
        log.info("[table_selector_llm] Model marked question out of scope: %r", question)
        return (
            [], None,
            "I can only answer questions related to the MES system. Please ask about work orders, BOMs, serial numbers, production, quality, or other manufacturing topics.",
            False, confidence,
        )

    if not picked:
        log.warning("[table_selector_llm] Model returned no valid tables — failing the request")
        return (
            [], None,
            "Couldn't find the relevant tables for this question.",
            in_scope, confidence,
        )


    if len(picked) > _TABLE_SELECTOR_MAX_TABLES:
        log.warning(
            "[table_selector_llm] Model picked %d table(s) (%s) — exceeds max of %d, "
            "truncating to the first %d",
            len(picked), picked, _TABLE_SELECTOR_MAX_TABLES, _TABLE_SELECTOR_MAX_TABLES,
        )
        _vprint("TABLESEL", "yellow",
                f" Picked {len(picked)} table(s), truncating to max "
                f"{_TABLE_SELECTOR_MAX_TABLES}: {picked[:_TABLE_SELECTOR_MAX_TABLES]}")
        picked = picked[:_TABLE_SELECTOR_MAX_TABLES]

    llm_intent: Optional[str] = raw_intent if raw_intent in _TABLE_SELECTOR_INTENT_LABELS else None
    if raw_intent and llm_intent is None:
        log.warning(
            "[table_selector_llm] Model returned unrecognised intent %r — ignoring, "
            "downstream will fall back to heuristic/SQL-gen classification.", raw_intent,
        )

    _vprint("TABLESEL", "cyan",
            f" table selection: {picked}",
            f"intent={llm_intent!r} module={module_name}/{subgroup_name} confidence={confidence}")

    return picked, llm_intent, None, in_scope, confidence


def _find_direct_fk_join(base: str, targets: list, all_tables: Optional[List[str]] = None) -> Optional[str]:
    _JP = _JD.get("join_paths", {})
    clause = base
    used: set = {base.lower()}

    for tgt in targets:
        if tgt.lower() == base.lower():
            continue

        join_step: Optional[dict] = None

        # Strategy 1: explicit join_paths entry (direct, depth-1)
        direct = _JP.get(base, {}).get(tgt)
        if direct and len(direct) == 1:
            join_step = direct[0]
            log.info(
                "[build_join_clause] Direct FK path found in join_paths: %s → %s via %s.%s",
                base, tgt, join_step.get("from_column"), join_step.get("to_column"),
            )

        # Strategy 2: shared column names (implicit FK detection)
        if join_step is None:
            base_cols = {c.lower(): c for c in META.get(base, {}).get("columns", {}).keys()}
            tgt_cols = {c.lower(): c for c in META.get(tgt, {}).get("columns", {}).keys()}
            shared = set(base_cols.keys()) & set(tgt_cols.keys())

            _INFRA_COLS = _SYSTEM_STATUS_COLS | {c.lower() for c in _AUDIT_COL_SUFFIXES}

            fk_candidates = {
                c for c in shared
                if c not in _INFRA_COLS and c.endswith("_id")
            }

            _prefix_residuals: set = {""}
            for _pfx in _TABLE_NAME_PREFIXES:
                for _seg in re.split(r"[_\s]+", _pfx.lower()):
                    if len(_seg) <= 2:
                        _prefix_residuals.add(_seg)

            def _fk_score(col: str) -> int:
                score = 0
                tgt_clean = tgt.lower()
                base_clean = base.lower()
                for pfx in _TABLE_NAME_PREFIXES:
                    tgt_clean = tgt_clean.replace(pfx.lower(), "")
                    base_clean = base_clean.replace(pfx.lower(), "")
                tgt_words = set(re.split(r"[_\s]+", tgt_clean)) - _prefix_residuals
                base_words = set(re.split(r"[_\s]+", base_clean)) - _prefix_residuals
                col_words = set(re.split(r"[_\s]+", col)) - {"", "id"}
                if col_words & tgt_words:
                    score += 3
                if col_words & base_words:
                    score += 2
                if any(col.endswith(sfx) for sfx in _COL_WORKFLOW_TYPE_SUFFIXES):
                    score += 1
                return score

            chosen = None
            if fk_candidates:
                best = max(fk_candidates, key=_fk_score)
                if _fk_score(best) > 0:
                    chosen = best
                else:
                    log.info(
                        "[build_join_clause] All shared _ID columns score 0 for %s <-> %s "
                        "-- no table-name overlap; falling back to BFS",
                        base, tgt,
                    )

            if chosen:
                _join_context = set(all_tables) if all_tables else {base, tgt}
                _has_master_anchor = any(_is_master_table(t) for t in _join_context)
                _both_link = (
                    not _is_master_table(base)
                    and not _is_master_table(tgt)
                    and not _has_master_anchor
                )
                if _both_link:
                    log.info(
                        "[build_join_clause] Strategy 2 blocked L/T→L/T relay: "
                        "%s <-> %s via '%s' — both non-master tables, "
                        "no master anchor in join context, would fan-out; forcing BFS fallback",
                        base, tgt, chosen,
                    )
                    chosen = None

            if chosen:
                join_step = {
                    "from_table": base,
                    "from_column": base_cols[chosen],
                    "to_table": tgt,
                    "to_column": tgt_cols[chosen],
                }
                log.info(
                    "[build_join_clause] Shared FK column detected: %s <-> %s via '%s' (score=%d)",
                    base, tgt, chosen, _fk_score(chosen),
                )

        if join_step is None:
            return None

        tt = join_step["to_table"]
        if tt.lower() not in used:
            clause += (
                f" INNER JOIN {tt}"
                f" ON {join_step['from_table']}.{join_step['from_column']}"
                f" = {tt}.{join_step['to_column']}"
            )
            used.add(tt.lower())

    return clause if "JOIN" in clause.upper() else None

def get_neighbor_tables(owner_table: str, hops: int = 2) -> List[str]:
    """Neo4j primary, BFS over join_paths as fallback."""
    try:
        neo4j_neighbors: List[str] = _neo4j_get_neighbor_tables(owner_table, hops=hops)
        if neo4j_neighbors:
            log.info(
                "[get_neighbor_tables] Neo4j: %r → %d neighbor(s) at hops=%d: %s",
                owner_table, len(neo4j_neighbors), hops, neo4j_neighbors,
            )
            _vprint("GRAPH", "green",
                    f" Neo4j neighbors ({hops}-hop): {owner_table!r}",
                    f"→ {neo4j_neighbors}")
            return neo4j_neighbors
        log.info(
            "[get_neighbor_tables] Neo4j returned empty for %r (hops=%d) — falling back to BFS",
            owner_table, hops,
        )
    except Exception as _neo4j_exc:
        log.warning(
            "[get_neighbor_tables] Neo4j unavailable for %r (%s) — falling back to BFS",
            owner_table, _neo4j_exc,
        )

    # BFS fallback
    _JP = _JD.get("join_paths", {})
    visited: set = {owner_table.lower()}
    frontier: List[str] = [owner_table]
    bfs_neighbors: List[str] = []
    for _ in range(hops):
        next_frontier: List[str] = []
        for tbl in frontier:
            for neighbor in _JP.get(tbl, {}).keys():
                if neighbor.lower() not in visited:
                    visited.add(neighbor.lower())
                    bfs_neighbors.append(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    log.info(
        "[get_neighbor_tables] BFS fallback: %r → %d neighbor(s) at hops=%d: %s",
        owner_table, len(bfs_neighbors), hops, bfs_neighbors,
    )
    _vprint("GRAPH", "yellow",
            f" BFS neighbor fallback ({hops}-hop): {owner_table!r}",
            f"→ {bfs_neighbors}")
    return bfs_neighbors


def _is_master_table(t: str) -> bool:
    """
    True if table t belongs to a 'master' tier (non-transaction prefix).
    Driven by TABLE_NAME_PREFIXES — excludes any prefix ending in _T_ or _T.
    """
    master_pfxs = tuple(
        p for p in _TABLE_NAME_PREFIXES
        if not any(p.upper().endswith(s) for s in ("_T_", "_T"))
    )
    return any(t.upper().startswith(p.upper()) for p in master_pfxs)


def _relay_reason(path: List[dict]) -> Optional[str]:
    """
    Return a rejection reason string, or None if the path is valid.

    Architecture rule: schemas follow master → link → transaction design.
    Any path that touches at least one master table is valid —
    the master provides the translation anchor that prevents row fan-out.
    Fan-out relay checks only apply to paths entirely within non-master
    (link/transaction) tables with no master table anywhere in the path.

    NOTE: callers in build_join_clause already guard with _path_has_master
    before calling this function, so this is a secondary safety net for
    any direct calls from other code paths.
    """
    all_tables = {step["from_table"] for step in path} | {step["to_table"] for step in path}
    if any(_is_master_table(t) for t in all_tables):
        return None

    for i, step in enumerate(path):
        fc = step["from_column"]
        tc = step["to_column"]
        ft = step["from_table"]
        tt = step["to_table"]

        if fc != tc:
            continue

        # Pattern 1: single-hop L/T→L/T relay
        both_non_master = not _is_master_table(ft) and not _is_master_table(tt)
        if both_non_master:
            return (
                f"L/T→L/T fan-out relay at hop {i}: "
                f"{ft}.{fc} → {tt}.{tc} "
                f"(same column both sides, no master table in path — "
                f"would multiply rows without a caller-supplied filter)"
            )

        # Pattern 2: multi-hop pass-through relay
        if i + 1 < len(path) and path[i + 1]["from_column"] == fc:
            return (
                f"Multi-hop pass-through relay at hop {i}: "
                f"{ft}.{fc} → {tt}.{tc} → "
                f"{path[i+1]['to_table']}.{path[i+1]['from_column']} "
                f"(key '{fc}' never narrowed across consecutive hops, no master anchor in path)"
            )

    return None

def build_join_clause(base: str, tables: list, intent: str = "simple") -> str:
    targets = [t for t in tables if t.lower() != base.lower()]
    if not targets:
        return base

    # Tier 1: direct FK detection
    direct_clause = _find_direct_fk_join(base, targets, all_tables=tables)
    if direct_clause:
        log.info("[build_join_clause] Tier 1 (direct FK) succeeded → %s", direct_clause[:120])
        _vprint("GRAPH", "green", " Direct FK join built", direct_clause[:100])
        return direct_clause

    log.info(
        "[build_join_clause] No direct FK for all targets — falling back to BFS. targets=%s", targets
    )

    # Tier 2: BFS over join_paths
    _JP = _JD.get("join_paths", {})
    clause, used = base, {base.lower()}
    all_bfs_ok = True

    for tgt in targets:
        path = _JP.get(base, {}).get(tgt)
        if path is None:
            q: deque = deque([(base, [])])
            visited = {base.lower()}
            while q:
                cur, acc = q.popleft()
                if len(acc) >= _JOIN_MAX_BFS_DEPTH:
                    continue
                for tt, steps in _JP.get(cur, {}).items():
                    new_acc = acc + [steps]
                    if tt.lower() == tgt.lower():
                        path = [s for seg in new_acc for s in seg]
                        break
                    if tt.lower() not in visited:
                        visited.add(tt.lower())
                        q.append((tt, new_acc))
                if path:
                    break

        if path is None:
            all_bfs_ok = False
            log.info(
                "[build_join_clause] BFS could not connect %r → %r — will try GraphJoinResolver",
                base, tgt,
            )
            break
        _path_tables = {s["from_table"] for s in path} | {s["to_table"] for s in path}
        _path_has_master = any(_is_master_table(t) for t in _path_tables)

        _relay = None
        if not _path_has_master:
            _relay = _relay_reason(path)

        if _relay:
            all_bfs_ok = False
            log.info(
                "[build_join_clause] BFS path for %r → %r rejected (relay, no master anchor): %s",
                base, tgt, _relay,
            )
            _vprint("GRAPH", "red",
                    f" BFS relay rejected: {base!r} → {tgt!r}",
                    _relay[:120])
            break

        for step in path:
            tt = step["to_table"]
            if tt.lower() not in used:
                clause += (
                    f" INNER JOIN {tt}"
                    f" ON {step['from_table']}.{step['from_column']}"
                    f" = {tt}.{step['to_column']}"
                )
                used.add(tt.lower())

    if all_bfs_ok and "JOIN" in clause.upper():
        log.info("[build_join_clause] Tier 2 (BFS) succeeded → %s", clause[:120])
        _vprint("GRAPH", "cyan", " BFS join built", clause[:100])
        return clause

    # Tier 3: GraphJoinResolver
    if _graph_resolver is not None:
        log.info("[build_join_clause] Tier 3 — delegating to GraphJoinResolver")
        _vprint("GRAPH", "yellow", " BFS incomplete — delegating to GraphJoinResolver")
        return _graph_resolver.build_join_clause(base, tables, intent)
    return clause


# ── Startup: type mappings + literal resolver FK map ─────────────────────────
try:
    with get_db_connection() as _startup_conn:
        load_type_mappings(_startup_conn, SCHEMA)
    log.info("[startup] Type mappings loaded.")
except Exception as _te:
    log.warning(f"[startup] Type mappings failed (non-fatal): {_te}")

_literal_resolver_init(META)
log.info("[startup] LiteralResolver FK map initialised.")

_TABLE_COL_WORDS: Dict[str, set] = {}
_COL_WORD_STOPWORDS = _COL_WORD_STOPWORDS_CFG

def _build_table_col_words() -> Dict[str, set]:
    cache: Dict[str, set] = {}
    for table, tinfo in META.items():
        words: set = set()
        for col in tinfo.get("columns", {}).keys():
            for tok in re.split(r'[_\s]+', col.lower()):
                if len(tok) >= 3 and tok not in _COL_WORD_STOPWORDS:
                    words.add(tok)
        cache[table] = words
    log.info(f"[startup] Column-word cache: {len(cache)} tables.")
    return cache

_TABLE_COL_WORDS.update(_build_table_col_words())

def _build_table_document(table_meta: dict) -> str:
    parts: List[str] = []
    name = table_meta.get("table_name", "")
    if name:
        human = name
        for _pfx in _TABLE_NAME_PREFIXES:
            human = human.replace(_pfx, "")
        human = human.replace("_", " ").lower()
        parts.append(human)
    for field in ("business_purpose", "row_represents"):
        val = table_meta.get(field, "")
        if val:
            parts.append(val)
    synonyms: List[str] = table_meta.get("synonyms", [])
    if synonyms:
        parts.append(". ".join(synonyms))
    use_when: List[str] = table_meta.get("use_when", [])
    if use_when:
        parts.append(". ".join(use_when))
    do_not_use: List[str] = table_meta.get("do_not_use_when", [])
    if do_not_use:
        parts.append("Not for: " + ". ".join(do_not_use))
    return " ".join(parts)

def _build_table_embeddings(embedder: SentenceTransformer) -> None:
    EmbeddingStore.instance().build_table_embeddings(embedder, META)


# ── Date helpers ──────────────────────────────────────────────────────────────
def _get_all_date_cols(table: str) -> set:
    _EFFECTIVE_MARKERS = {"effective", "valid", "active", "eff"}
    _BOUND_MARKERS = {"from", "to", "start", "end", "until", "since"}
    result = set()
    for c in META.get(table, {}).get("columns", {}).keys():
        cl = c.lower()
        parts = set(re.split(r'[_\s]+', cl))
        if "date" in cl or "time" in cl or "timestamp" in cl:
            result.add(c)
            continue
        if parts & _EFFECTIVE_MARKERS and parts & _BOUND_MARKERS:
            result.add(c)
            continue
        if "cdate" in cl or "udate" in cl or "ctime" in cl or "utime" in cl:
            result.add(c)
    return result

def _col_intent(col_name: str) -> str:
    parts = set(re.split(r'[_\s]+', col_name.lower()))
    _EFFECTIVE_MARKERS = {"effective", "valid", "active", "eff"}
    _BOUND_MARKERS = {"from", "to", "start", "end", "until", "since"}
    if parts & _EFFECTIVE_MARKERS:
        return "effective"
    if parts & {"created", "create", "cdate", "entry", "entered", "inserted",
                "ctime", "creation_date", "open", "raised", "registered"}:
        return "creation"
    if parts & {"modified", "updated", "udate", "changed", "utime", "last_modified",
                "last_updated", "edit", "edited"}:
        return "modified"
    return "other"

def _date_cols_in_where(sql: str, table: str) -> List[str]:
    all_date_cols = _get_all_date_cols(table)
    try:
        expr = sqlglot.parse_one(
            sql, dialect="tsql",
            error_level=_sqlglot_errors.ErrorLevel.IGNORE,
        )
        where = expr.find(sqlglot.exp.Where)
        if not where:
            return []
        referenced = {
            col.name.lower()
            for col in where.find_all(sqlglot.exp.Column)
        }
        return [c for c in all_date_cols if c.lower() in referenced]
    except Exception:
        return []

def find_date_column(table: str, prefer_creation: bool = True) -> Optional[str]:
    """
    Return the best date column to inject a time filter on.

    Priority order:
      1. Exact match in _DATE_PRIORITY_SUFFIXES (e.g. "cdate", "created_date")
      2. Suffix match in _DATE_PRIORITY_SUFFIXES (e.g. "wo_created_date")
      3. Any column whose name contains "date" or "time" (non-effective)
      4. Any remaining date/time column (including effective ones)
      5. Fallback — any column recognized by _get_all_date_cols() (e.g.
         "INT_Valid_From"), which catches effective/validity columns that
         don't literally contain "date" or "time" in their name. Without
         this step, such columns are correctly stripped by
         strip_llm_date_filters() but then never get a replacement filter
         injected, silently leaving the query unfiltered.

    If prefer_creation=True (default), effective-date columns are deprioritised
    to step 4 so that transaction queries filter on creation date, not validity.
    """
    cols = META.get(table, {}).get("columns", {})
    for p in _DATE_PRIORITY_SUFFIXES:
        for c in cols:
            if c.lower() == p or c.lower().endswith("_" + p):
                return c
    _EFF_MARKERS = {"effective", "valid", "eff", "active"}
    for c in cols:
        cl = c.lower()
        if ("date" in cl or "time" in cl):
            if prefer_creation and any(m in cl for m in _EFF_MARKERS):
                continue
            return c
    for c in cols:
        if "date" in c.lower() or "time" in c.lower():
            return c
    fallback_cols = _get_all_date_cols(table)
    if fallback_cols:
        non_effective = [c for c in fallback_cols if _col_intent(c) != "effective"]
        chosen = sorted(non_effective) if non_effective else sorted(fallback_cols)
        if chosen:
            if not non_effective:
                log.warning(
                    "[find_date_column] No creation-date column found for %r — "
                    "falling back to effective/validity column %r",
                    table, chosen[0],
                )
            return chosen[0]

    return None

def inject_effective_filter(sql: str, table: str, question: str) -> str:
    if _dr_has_current_intent(question):
        log.info("[inject_effective] Skipping — current/now intent detected; effective filter not applicable.")
        return sql

    cols_in_where = _date_cols_in_where(sql, table)
    if cols_in_where:
        intents = {_col_intent(c) for c in cols_in_where}
        if "effective" in intents:
            log.info("[inject_effective] Skipping — effective col already in WHERE: %r", cols_in_where)
            return sql
        if "creation" in intents:
            log.info("[inject_effective] Skipping — creation col in WHERE: %r", cols_in_where)
            return sql

    sql_tables = _extract_sql_tables(sql)
    use_table = sql_tables[0] if sql_tables else table
    cols = list(META.get(use_table, {}).get("columns", {}).keys())
    cols_lower = {col.lower(): col for col in cols}
    eff_from = next(
        (cols_lower[col] for col in cols_lower
         if "effective" in col and ("from" in col or "start" in col or col.endswith("_from"))),
        None,
    )
    eff_to = next(
        (cols_lower[col] for col in cols_lower
         if "effective" in col and ("to" in col or "end" in col or col.endswith("_to"))),
        None,
    )
    if not eff_from or not eff_to:
        return sql
    return _dr_inject_effective(sql, use_table, question, eff_from, eff_to, tz_name=USER_TIMEZONE)

def inject_time_filter(sql: str, table: str, question: str) -> str:
    """
    Inject a date/time WHERE filter onto the SQL.

    This function is only called when has_temporal_intent(question) is True,
    meaning the user EXPLICITLY mentioned a date or time reference.
    """
    cols_in_where = _date_cols_in_where(sql, table)
    if cols_in_where:
        intents = {_col_intent(c) for c in cols_in_where}
        if "effective" in intents:
            log.info("[inject_time] Skipping — effective col already in WHERE: %r", cols_in_where)
            return sql
        if "creation" in intents:
            if not _dr_has_current_intent(question):
                log.info("[inject_time] Skipping — creation col already in WHERE: %r", cols_in_where)
                return sql
    _skip_for_agg = False
    if not _dr_has_current_intent(question):
        try:
            _guard_expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
            _skip_for_agg = _guard_expr.find(sqlglot.exp.Group) is not None
        except Exception:
            _skip_for_agg = False

    if _skip_for_agg:
        log.info("[inject_time] Skipping — SQL has GROUP BY (non-current query).")
        return sql

    sql_tables = _extract_sql_tables(sql)
    actual_table = sql_tables[0] if sql_tables else table
    use_table = actual_table if find_date_column(actual_table) else table
    dc = find_date_column(use_table)
    if not dc:
        return sql
    log.info(f"[inject_time] {use_table!r} → {use_table}.{dc}")
    sql = inject_date_filter(
        sql, use_table, question, dc,
        all_date_cols=_get_all_date_cols(use_table),
        tz_name=USER_TIMEZONE,
    )
    if _parse_date_range(question, tz_name=USER_TIMEZONE) is None:
        sql = _dr_inject_time_filter(sql, use_table, question, dc)
    return sql


# ── SQL utilities ─────────────────────────────────────────────────────────────
def _extract_sql_tables(sql: str) -> List[str]:
    known = {t.lower(): t for t in META.keys()}
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)
        tables = []
        for t in expr.find_all(sqlglot.exp.Table):
            name = t.name
            if name.lower() in known:
                tables.append(known[name.lower()])
        return list(dict.fromkeys(tables))
    except Exception:
        log.warning("[extract_sql_tables] sqlglot parse failed — returning empty table list")
        return []


def convert_mysql_to_sqlserver(sql: str) -> str:
    if "LIMIT " not in sql.upper():
        return sql
    sql = re.sub(r'\bLIMIT\s+\d+\s*;?\s*$', '', sql, flags=re.IGNORECASE).rstrip()
    log.info("[convert_mysql] Stripped trailing LIMIT clause")
    return sql


_PLACEHOLDER_TOKEN_RE = re.compile(r"<\s*[A-Za-z0-9_]+\s*>")


def scrub_invalid_sql_tokens(sql: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    sql = ansi_escape.sub('', sql)
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.I)
    sql = re.sub(r"```", "", sql)
    sql = re.sub(r"--[^\n]*", "", sql)
    if _PLACEHOLDER_TOKEN_RE.search(sql):
        before = sql
        sql = re.sub(
            r"(?:\b(?:AND|OR)\s+)?[\w.\[\]]+\s*(?:=|<>|!=|>=|<=)\s*"
            + _PLACEHOLDER_TOKEN_RE.pattern
            + r"(?:\s+(?:AND|OR)\b)?",
            " ",
            sql,
            flags=re.I,
        )
        sql = _PLACEHOLDER_TOKEN_RE.sub("NULL", sql)
        if sql != before:
            log.warning("[scrub] LLM placeholder token(s) detected — stripped/neutralized")

    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\bWHERE\s+(?:AND|OR)\b", "WHERE", sql, flags=re.I)
    sql = re.sub(r"\bWHERE\s*(?=ORDER\s+BY|GROUP\s+BY|$)", "", sql, flags=re.I)
    sql = re.sub(r"\s+(?:AND|OR)\s+(?=ORDER\s+BY|GROUP\s+BY|$)", " ", sql, flags=re.I)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def _enclosing_select(node) -> Optional["sqlglot.exp.Select"]:
    """Nearest ancestor SELECT of `node` (not counting `node` itself)."""
    p = node.parent
    while p is not None and not isinstance(p, sqlglot.exp.Select):
        p = p.parent
    return p


def _direct_from_join_tables(select: "sqlglot.exp.Select") -> List["sqlglot.exp.Table"]:
    """Table nodes directly in `select`'s own FROM/JOIN — never descends
    into a nested subquery's SELECT, since that subquery's tables belong
    to ITS scope, not this one. This is the key fix: the old code used
    expr.find_all(Table), which walks the ENTIRE tree including tables
    buried inside unrelated subqueries, letting a table that only exists
    inside one subquery's FROM get treated as a globally valid qualifier
    anywhere else in the query — a silent cross-scope leak that is exactly
    how a dangling "TABLE.col" reference could pass validation here and
    still blow up as "multipart identifier could not be bound" in SQL Server."""
    out: List["sqlglot.exp.Table"] = []
    candidates = []
    from_node = select.args.get("from")
    if from_node is not None:
        candidates.append(from_node.this)
    for j in (select.args.get("joins") or []):
        candidates.append(j.this)
    for node in candidates:
        if isinstance(node, sqlglot.exp.Table):
            out.append(node)
    return out


def _direct_subquery_aliases(select: "sqlglot.exp.Select") -> set:
    """Aliases of derived-table subqueries directly in this SELECT's own
    FROM/JOIN (e.g. FROM (SELECT ...) AS X). Valid qualifiers in this
    scope, but we can't verify their inner projected columns against a
    schema, so callers should treat them permissively."""
    out: set = set()
    candidates = []
    from_node = select.args.get("from")
    if from_node is not None:
        candidates.append(from_node.this)
    for j in (select.args.get("joins") or []):
        candidates.append(j.this)
    for node in candidates:
        if isinstance(node, sqlglot.exp.Subquery):
            alias = (node.alias_or_name or "").lower()
            if alias:
                out.add(alias)
    return out


class _SqlResolutionContext:
    """
    Single source of truth for "what tables/columns are actually visible
    from THIS scope", built once per SELECT (main query or a nested
    subquery) from that scope's own real FROM/JOIN clause — never from
    the caller's bookkeeping lists (join_tables/all_tables), which can
    drift out of sync with what actually ended up in the SQL, and never
    from unrelated sibling/nested subqueries, whose tables aren't visible
    here at all.

    Replaces five previously-independent implementations of "does this
    table qualifier belong here" (_rewrite_to_tablename_qualified,
    _sqlglot_validate_column_refs, _resolve_unknown_table_prefixes,
    _strip_orphaned_table_refs, _sqlglot_strip_orphaned_orderby) that each
    used slightly different, incomplete logic and could disagree with
    each other.
    """

    def __init__(
        self,
        expr,
        all_tables: List[str],
        live_cols: Optional[Dict[str, List[str]]] = None,
        scope_select: Optional["sqlglot.exp.Select"] = None,
    ):
        self.alias_to_table: Dict[str, str] = {}
        self.real_to_alias: Dict[str, str] = {}
        self.real_name_lookup: set = set()
        self.real_tables: set = set()
        self.subquery_aliases: set = set()
        self.cte_names: set = set(
            (c.alias_or_name or "").lower() for c in expr.find_all(sqlglot.exp.CTE)
        )
        target = scope_select or expr.find(sqlglot.exp.Select)
        node = target
        _depth_guard = 0
        while node is not None and _depth_guard < 25:
            _depth_guard += 1
            if isinstance(node, sqlglot.exp.Select):
                for tnode in _direct_from_join_tables(node):
                    real = (tnode.name or "").upper()
                    if not real:
                        continue
                    self.real_tables.add(real)
                    self.real_name_lookup.add(real.lower())
                    alias = (tnode.alias_or_name or "").lower()
                    if alias and alias != real.lower():
                      
                        self.alias_to_table[alias] = real
                        self.real_to_alias[real] = alias
                    else:
                        self.alias_to_table[real.lower()] = real
                self.subquery_aliases |= _direct_subquery_aliases(node)
            node = _enclosing_select(node)
        if target is None:
            for tnode in expr.find_all(sqlglot.exp.Table):
                real = (tnode.name or "").upper()
                if not real:
                    continue
                self.real_tables.add(real)
                self.alias_to_table[real.lower()] = real
                alias = (tnode.alias_or_name or "").lower()
                if alias and alias != real.lower():
                    self.alias_to_table[alias] = real
        self.table_columns: Dict[str, set] = {}
        scope_tables = self.real_tables or {t.upper() for t in all_tables}
        for tbl in scope_tables:
            if live_cols and live_cols.get(tbl):
                cols = live_cols[tbl]
            elif FULL_SCHEMA_CACHE.get(tbl, {}).get("columns"):
                cols = FULL_SCHEMA_CACHE[tbl]["columns"]
            else:
                cols = list(META.get(tbl, {}).get("columns", {}).keys())
            self.table_columns[tbl] = {c.lower() for c in cols}

        self.col_owner: Dict[str, List[str]] = {}
        for tbl, cols in self.table_columns.items():
            for c in cols:
                self.col_owner.setdefault(c, []).append(tbl)

    def resolve_owner(self, col_name: str) -> Optional[str]:
        """Returns the single owning table for a column name, or None if the
        column is unknown OR ambiguous (owned by >1 table). We never guess
        on ambiguity — a wrong guess produces silently-wrong data, whereas
        leaving it bare surfaces as a normal 'ambiguous column' DB error
        that the existing retry path already handles."""
        owners = self.col_owner.get((col_name or "").lower(), [])
        return owners[0] if len(owners) == 1 else None

    def qualifier_for(self, real_table: str) -> str:
        """The qualifier that's actually legal to write for this table in
        this scope: its alias if the FROM/JOIN gave it one, otherwise the
        real table name. Never return a real name for a table that's
        aliased here — SQL Server rejects that with 'multi-part identifier
        could not be bound' even though the table genuinely is in scope."""
        return self.real_to_alias.get((real_table or "").upper(), real_table)

    def column_exists(self, table: str, col_name: str) -> bool:
        cols = self.table_columns.get((table or "").upper())
        if cols is None:
            return True
        return (col_name or "").lower() in cols

def _resolve_columns(
    sql: str,
    all_tables: List[str],
    live_cols: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    One pass over every Column node in the query, resolved against the
    _SqlResolutionContext of its OWN enclosing SELECT scope (main query or
    whichever nested subquery it actually lives in) — not one flat context
    built from every Table node anywhere in the tree. That distinction
    matters: a table used only inside subquery B has no business being
    treated as a valid qualifier for a column that lives in subquery A, or
    in the outer query. Letting that cross-scope reference through was
    exactly how a dangling "TABLE.col" could survive this pass and still
    hit SQL Server as "multipart identifier could not be bound".

    Per column, by scope:
      - qualified with a real FROM/JOIN table or alias VISIBLE IN THIS SCOPE
        -> canonicalize to the real table name
      - qualified with a CTE name (global) or a derived-table subquery
        alias visible in this scope -> leave untouched
      - qualified with something that matches NEITHER -> this is exactly the
        "multipart identifier could not be bound" defect. Try to recover by
        column name against tables actually visible in this scope; if that's
        unresolvable or ambiguous, strip the qualifier bare rather than leave
        a dangling reference to a table that isn't there.
      - bare -> qualify only if the
        column name uniquely identifies one table visible in this scope;
        otherwise leave bare (ambiguous — let the DB's real "ambiguous
        column" error drive the existing LLM retry path instead of silently
        guessing).
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)

        select_nodes = list(expr.find_all(sqlglot.exp.Select))
        if not select_nodes:
            return sql

        ctx_by_select = {
            id(sel): _SqlResolutionContext(expr, all_tables, live_cols=live_cols, scope_select=sel)
            for sel in select_nodes
        }
        default_ctx = ctx_by_select[id(select_nodes[0])]

        _requalified: List[str] = []
        _stripped: List[str] = []
        _ambiguous_qualified: List[str] = []

        for col in expr.find_all(sqlglot.exp.Column):
            owning_select = _enclosing_select(col)
            ctx = ctx_by_select.get(id(owning_select), default_ctx) if owning_select else default_ctx

            if col.table:
                ref = col.table.lower()
                if ref in ctx.alias_to_table:
                    real = ctx.alias_to_table[ref]
                    qualifier = ctx.qualifier_for(real)
                    if col.table != qualifier:
                        col.set("table", sqlglot.exp.Identifier(this=qualifier))
                elif ref in ctx.real_name_lookup:
                   
                    qualifier = ctx.qualifier_for(ref.upper())
                    if col.table != qualifier:
                        col.set("table", sqlglot.exp.Identifier(this=qualifier))
                elif ref in ctx.cte_names or ref in ctx.subquery_aliases:
                    continue
                else:
                    owner = ctx.resolve_owner(col.name)
                    if owner:
                        qualifier = ctx.qualifier_for(owner)
                        log.warning(
                            "[resolve_columns] '%s.%s' — '%s' not visible in this scope's FROM/JOIN; "
                            "requalified to '%s.%s'",
                            ref, col.name, ref, qualifier, col.name,
                        )
                        _requalified.append(f"{ref}.{col.name} -> {qualifier}.{col.name}")
                        col.set("table", sqlglot.exp.Identifier(this=qualifier))
                    else:
                        log.warning(
                            "[resolve_columns] '%s.%s' — '%s' not visible in this scope's FROM/JOIN and "
                            "no unambiguous owner found; stripping qualifier bare",
                            ref, col.name, ref,
                        )
                        _stripped.append(f"{ref}.{col.name}")
                        col.set("table", None)
            else:
                # ── BARE COLUMN ──────────────────────────────────────────
                owner = ctx.resolve_owner(col.name)
                if owner:
                    col.set("table", sqlglot.exp.Identifier(this=ctx.qualifier_for(owner)))
                else:
                    owners = ctx.col_owner.get((col.name or "").lower(), [])
                    if len(owners) > 1:
                        chosen_table = _choose_ambiguous_table(
                            col.name, owners, all_tables, sql, ctx
                        )
                        if chosen_table:
                            qualifier = ctx.qualifier_for(chosen_table)
                            col.set("table", sqlglot.exp.Identifier(this=qualifier))
                            _ambiguous_qualified.append(f"{col.name} -> {qualifier}.{col.name}")
                            log.info(
                                "[resolve_columns] Ambiguous column '%s' auto-qualified to %s.%s",
                                col.name, qualifier, col.name,
                            )
                        else:
                            log.warning(
                                "[resolve_columns] Ambiguous column '%s' cannot be resolved — leaving bare",
                                col.name,
                            )

        if _requalified or _stripped or _ambiguous_qualified:
            log.warning(
                "[resolve_columns] summary — requalified=%s stripped_bare=%s ambiguous_qualified=%s",
                _requalified, _stripped, _ambiguous_qualified,
            )
            if _ambiguous_qualified:
                _vprint(
                    "ALIAS", "green",
                    f" Auto-qualified {len(_ambiguous_qualified)} ambiguous column(s)",
                    f"cols={_ambiguous_qualified}",
                )

        return expr.sql(dialect="tsql")
    except Exception as exc:
        log.error("[resolve_columns] sqlglot failed: %s", exc)
        return sql


def _choose_ambiguous_table(
    col_name: str,
    owners: List[str],
    all_tables: List[str],
    sql: str,
    ctx: "_SqlResolutionContext",
) -> Optional[str]:
    """
    Choose the most likely owning table for a bare column name that
    _SqlResolutionContext.resolve_owner() couldn't resolve because more
    than one visible table has a column with this name. Tries several
    data-driven strategies in order of reliability; returns the first
    match, or None if nothing applies (caller leaves the column bare, same
    as the old behavior, so this only ever adds resolution power — it
    never makes a previously-bare-and-fine case worse).
    """
    if not owners:
        return None

    # Strategy 1: prefer a table that's already qualifying THIS column name
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        from_tables = [t.name for t in expr.find_all(sqlglot.exp.Table)]
        for table in from_tables:
            if table in owners:
                log.info("[choose_ambiguous] Chose %s for %s (appears in FROM)", table, col_name)
                return table
    except Exception:
        pass

    # Strategy 2: prefer master tables over transaction/link tables
    for table in owners:
        if _is_master_table(table):
            log.info("[choose_ambiguous] Chose %s for %s (master table)", table, col_name)
            return table

    # Strategy 3: prefer a table where this column is its primary key.
    for table in owners:
        if col_name.lower() in _get_table_pk_cols(table):
            log.info("[choose_ambiguous] Chose %s for %s (PK column)", table, col_name)
            return table

    # Strategy 4: prefer a table where this column is used as the join key
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        for join in expr.find_all(sqlglot.exp.Join):
            on_expr = join.args.get("on")
            if not on_expr:
                continue
            for col_node in on_expr.find_all(sqlglot.exp.Column):
                if col_node.name.lower() == col_name.lower() and col_node.table in owners:
                    log.info("[choose_ambiguous] Chose %s for %s (FK in JOIN)", col_node.table, col_name)
                    return col_node.table
    except Exception:
        pass

    # Strategy 5: prefer whichever owner appears first in the FROM clause
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        from_node = expr.find(sqlglot.exp.From)
        if from_node:
            for table_node in from_node.find_all(sqlglot.exp.Table):
                table_name = table_node.name
                if table_name in owners:
                    log.info("[choose_ambiguous] Chose %s for %s (first in FROM)", table_name, col_name)
                    return table_name
    except Exception:
        pass

    # Strategy 6: safety check — confirm against live/schema column list.
    for table in owners:
        if ctx.column_exists(table, col_name):
            log.info("[choose_ambiguous] Chose %s for %s (column exists in schema)", table, col_name)
            return table

    log.warning("[choose_ambiguous] No strategy matched — using fallback first owner %s for %s", owners[0], col_name)
    return owners[0]

def _clean_order_and_group_by(sql: str, all_tables: List[str],
                               live_cols: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Validate every ORDER BY / GROUP BY item against the same resolution
    context used for SELECT/WHERE columns. Handles all three cases the old
    _sqlglot_strip_orphaned_orderby missed:
      1. bare (unqualified) order/group columns — now resolved, not ignored
      2. non-Column items (literal/constant, stale
         SELECT-alias reference) — validated, not ignored
      3. qualified-but-nonexistent column in a
         real table — now checked, not just the table name
    A non-integer constant or an unresolvable reference in ORDER BY is exactly
    what SQL Server's "constant expression was encountered in the ORDER BY
    list" fires on, so those get removed instead of forwarded to the DB.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)
        ctx = _SqlResolutionContext(expr, all_tables, live_cols=live_cols)

        select = expr.find(sqlglot.exp.Select)
        select_aliases = {
            e.alias.lower() for e in (select.expressions if select else [])
            if getattr(e, "alias", None)
        }
        n_select_cols = len(select.expressions) if select else 0

        def _resolve_ordered_item(item) -> bool:
            """Returns True if the item is valid (mutating it in place to
            canonicalize/qualify as needed), False if it should be dropped."""
            inner = item.this
            if isinstance(inner, sqlglot.exp.Column):
                if inner.table:
                    real = ctx.alias_to_table.get(inner.table.lower())
                    if real is None:
                        return False
                    if not ctx.column_exists(real, inner.name):
                        return False
                    if inner.table != real:
                        inner.set("table", sqlglot.exp.Identifier(this=real))
                    return True
                else:
                    if inner.name.lower() in select_aliases:
                        return True
                    owner = ctx.resolve_owner(inner.name)
                    if owner:
                        inner.set("table", sqlglot.exp.Identifier(this=owner))
                        return True
                    return False
            if isinstance(inner, sqlglot.exp.Literal) and not inner.is_string:
                try:
                    pos = int(inner.this)
                    return 1 <= pos <= n_select_cols
                except (TypeError, ValueError):
                    return False
            return False

        for order_clause in list(expr.find_all(sqlglot.exp.Order)):
            for order_item in list(order_clause.find_all(sqlglot.exp.Ordered)):
                if not _resolve_ordered_item(order_item):
                    log.warning("[clean_order_by] Removing invalid ORDER BY item: %s",
                                order_item.sql(dialect="tsql"))
                    order_item.pop()
            if not order_clause.expressions:
                log.warning("[clean_order_by] ORDER BY left with no valid items — removing clause")
                order_clause.pop()

        for grp in list(expr.find_all(sqlglot.exp.Group)):
            for col in list(grp.find_all(sqlglot.exp.Column)):
                real = ctx.alias_to_table.get(col.table.lower()) if col.table else ctx.resolve_owner(col.name)
                if not real or not ctx.column_exists(real, col.name):
                    log.warning("[clean_group_by] Removing invalid GROUP BY col: %s",
                                col.sql(dialect="tsql"))
                    col.pop()
                elif col.table != real:
                    col.set("table", sqlglot.exp.Identifier(this=real))
            if not grp.expressions:
                log.warning("[clean_group_by] GROUP BY left with no valid columns — removing clause")
                grp.pop()

        return expr.sql(dialect="tsql")
    except Exception as exc:
        log.warning(f"[clean_order_and_group_by] failed: {exc}")
        return sql


def _fix_group_by_coherence(sql: str) -> str:
    """
    Ensure every non-aggregated SELECT column appears in GROUP BY.
    If GROUP BY exists and SELECT has bare columns not in it, add them.
    If no GROUP BY but aggregates are mixed with bare columns, synthesize one.
    No-op when there are no aggregate functions in the SELECT.
    """
    _AGG_TYPES = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max, sqlglot.exp.Anonymous,
    )
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql",
                                 error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql

        has_agg = any(sel_item.find(*_AGG_TYPES) for sel_item in select.expressions)
        if not has_agg:
            return sql

        group = expr.find(sqlglot.exp.Group)
        bare_cols = []
        for sel_expr in select.expressions:
            inner = sel_expr.this if isinstance(sel_expr, sqlglot.exp.Alias) else sel_expr
            if inner.find(*_AGG_TYPES):
                continue
            if isinstance(inner, (sqlglot.exp.Column, sqlglot.exp.Identifier)):
                bare_cols.append(inner.copy())
        if not bare_cols:
            return sql
        if group is None:

            log.warning(
                "[fix_group_by_coherence] No GROUP BY despite aggregates — synthesizing from %d bare col(s)",
                len(bare_cols),
            )
            group_expr = sqlglot.exp.Group(expressions=bare_cols)
            expr.set("group", group_expr)
        else:
            existing_keys: set = set()
            for gb_col in group.find_all(sqlglot.exp.Column):
                existing_keys.add(
                    (gb_col.table.lower() if gb_col.table else "", gb_col.name.lower())
                )
            for gb_id in group.find_all(sqlglot.exp.Identifier):
                existing_keys.add(("", gb_id.name.lower()))

            missing = []
            for col in bare_cols:
                tbl = (col.table.lower() if hasattr(col, "table") and col.table else "")
                nm = col.name.lower() if hasattr(col, "name") else ""
                if (tbl, nm) not in existing_keys and ("", nm) not in existing_keys:
                    missing.append(col)

            if missing:
                log.warning(
                    "[fix_group_by_coherence] Adding %d missing col(s) to GROUP BY: %s",
                    len(missing),
                    [
                        f"{c.table}.{c.name}" if hasattr(c, "table") and c.table else getattr(c, "name", str(c))
                        for c in missing
                    ],
                )
                group.set("expressions", list(group.expressions) + missing)

        return expr.sql(dialect="tsql")
    except Exception as exc:
        log.warning("[fix_group_by_coherence] Failed: %s", exc)
        return sql

def _sqlglot_replace_select_star(sql: str, tables: List[str], question: str, intent: str = "simple") -> str:
    """
    Column-selection plane removed. This used to expand SELECT * into an
    explicit narrowed column list via _auto_select_columns_multi(); that
    logic is gone, so SELECT * is now left untouched.
    """
    return sql


def _fix_all_string_literals(sql: str, all_tables: List[str]) -> str:
    """
    Single-parse replacement for running _uppercase_string_literals() and
    _fix_numeric_col_string_literal_in_sql() as two separate sqlglot
    parse/walk/serialize passes back to back. Both passes only care about
    `<Column> = <string literal>` equality nodes, so they're folded into
    one AST walk here: varchar columns get UPPER()-wrapped for
    case-insensitive comparison, numeric columns get their mismatched
    string literal resolved to the real stored value (LOV / FK / reverse
    lookup / brute-force, same resolution order as before).
    """
    _VARCHAR_TYPES = {"varchar", "nvarchar", "char", "nchar", "text", "ntext", "string"}
    _NUMERIC_TYPES = {
        "int", "integer", "bigint", "smallint", "tinyint",
        "numeric", "decimal", "float", "real", "number",
    }

    varchar_cols: set = set()
    numeric_col_table: Dict[str, str] = {}
    for t in all_tables:
        for col, cinfo in META.get(t, {}).get("columns", {}).items():
            dtype = (cinfo.get("data_type") or "" if isinstance(cinfo, dict) else "").lower()
            if dtype in _VARCHAR_TYPES:
                varchar_cols.add(col.lower())
            elif any(dtype.startswith(nt) for nt in _NUMERIC_TYPES):
                numeric_col_table[col.lower()] = t

    if not varchar_cols and not numeric_col_table:
        return sql

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
    except Exception as exc:
        log.warning(f"[sqlglot] _fix_all_string_literals parse failed: {exc}")
        return sql

    changed = False

    def _resolve_numeric(col_lower: str, display_val: str, owning_table: str) -> Optional[str]:
        dv_up = display_val.upper()
        with _reverse_lookup_lock:
            rl_idx = _REVERSE_LOOKUP_INDEX
        dv_lower = display_val.lower()
        if dv_lower in rl_idx:
            for entry in rl_idx[dv_lower]:
                if entry["pk_col"].lower() == col_lower:
                    return entry["id"]
            if len(rl_idx[dv_lower]) == 1:
                return rl_idx[dv_lower][0]["id"]

        # 1. LOV_DATA
        for tbl in all_tables:
            for lov_col, mapping in LOV_DATA.get(tbl, {}).items():
                if lov_col.lower() == col_lower:
                    for _raw_key, _meta in mapping.items():
                        label, db_val = _lov_label_value(_raw_key, _meta)
                        if label.upper() == dv_up or str(_raw_key).upper() == dv_up:
                            return db_val

        # 2. FK map
        fk_map = get_fk_map()
        for fk_entry in fk_map.values():
            master_tbl = fk_entry.get("master_table", "")
            name_col = fk_entry.get("name_col", "")
            pk_col = fk_entry.get("master_pk", "")
            fk_col = fk_entry.get("fk_col", "").lower()
            if not (master_tbl and name_col and pk_col):
                continue
            if col_lower not in (fk_col, pk_col.lower()):
                continue
            try:
                with get_db_connection() as _c:
                    _cur = _c.cursor()
                    _cur.execute(
                        f"SELECT TOP 1 [{pk_col}] FROM [{master_tbl}] WITH (NOLOCK) "
                        f"WHERE UPPER([{name_col}]) = ?", (dv_up,)
                    )
                    _row = _cur.fetchone()
                    if _row:
                        return str(_row[0])
            except Exception as _e:
                log.warning("[fix_numeric_literal] FK lookup failed: %s", _e)

        # 3. Brute-force: try varchar name/desc/code cols in the owning table
        varchar_types = {"varchar", "nvarchar", "char", "nchar", "text", "ntext"}
        name_keywords = ("name", "desc", "code", "title", "no", "number")
        try:
            tbl_cols = META.get(owning_table, {}).get("columns", {})
            candidates = [
                c for c, ci in tbl_cols.items()
                if isinstance(ci, dict)
                and (ci.get("data_type") or "").lower() in varchar_types
                and any(kw in c.lower() for kw in name_keywords)
            ]
            for name_col in candidates[:5]:
                try:
                    with get_db_connection() as _c:
                        _cur = _c.cursor()
                        _cur.execute(
                            f"SELECT TOP 1 [{col_lower}] FROM [{owning_table}] WITH (NOLOCK) "
                            f"WHERE UPPER([{name_col}]) = ?", (dv_up,)
                        )
                        _row = _cur.fetchone()
                        if _row:
                            log.info("[fix_numeric_literal] Brute-force: %s.%s='%s' -> %s=%s",
                                     owning_table, name_col, display_val, col_lower, _row[0])
                            return str(_row[0])
                except Exception:
                    continue
        except Exception as _e:
            log.warning("[fix_numeric_literal] Brute-force failed: %s", _e)
        return None

    for eq in expr.find_all(sqlglot.exp.EQ):
        left, right = eq.left, eq.right
        if isinstance(left, sqlglot.exp.Upper):
            continue
        if not isinstance(left, sqlglot.exp.Column):
            continue
        if not isinstance(right, sqlglot.exp.Literal) or not right.is_string:
            continue

        col_name = left.name.lower()

        if col_name in numeric_col_table:
            display_val = right.this
            owning_table = numeric_col_table[col_name]
            log.warning("[fix_numeric_literal] Numeric col '%s' compared to string '%s' — resolving",
                        col_name, display_val)
            _vprint("VALIDATE", "yellow",
                    f" Type mismatch: '{col_name}' is numeric, literal='{display_val}' — resolving ID")
            resolved = _resolve_numeric(col_name, display_val, owning_table)
            if resolved is not None:
                eq.set("right",
                       sqlglot.exp.Literal.number(resolved)
                       if resolved.lstrip("-").isdigit()
                       else sqlglot.exp.Literal.string(resolved))
                changed = True
                log.info("[fix_numeric_literal] '%s' -> %s for col '%s'", display_val, resolved, col_name)
                _vprint("VALIDATE", "green",
                        f" Type fix: '{col_name}' = '{display_val}' → {resolved}")
            else:
                log.error("[fix_numeric_literal] Cannot resolve '%s' for numeric col '%s'",
                          display_val, col_name)
            continue

        if col_name in varchar_cols:
            left_copy = left.copy()
            upper = sqlglot.exp.Upper(this=left_copy)
            eq.set("left", upper)
            right.set("this", right.this.upper())
            changed = True

    if not changed:
        return sql
    try:
        return expr.sql(dialect="tsql")
    except Exception:
        return sql

def _strip_status_filters_in_ast(expr: "sqlglot.exp.Expression", question: str) -> bool:
    """
    AST-mutating replacement for _sqlglot_strip_spurious_status_filter().
    The main post-processing pass now calls this directly on its own
    already-parsed `expr`, right after the single up-front parse, instead
    of round-tripping through its own parse -> print -> (caller reparses)
    cycle. Mutates `expr` in place; returns True if a predicate was
    dropped.
    """
    if _question_implies_system_status(question) or not _SYSTEM_STATUS_COLS:
        return False
    where = expr.find(sqlglot.exp.Where)
    if not where:
        return False

    def _collect_predicates(node) -> List[sqlglot.exp.Expression]:
        if isinstance(node, sqlglot.exp.And):
            return _collect_predicates(node.left) + _collect_predicates(node.right)
        return [node]

    predicates = _collect_predicates(where.this)
    kept = []
    dropped = False
    for pred in predicates:
        col = pred.find(sqlglot.exp.Column)
        if col and col.name.lower() in _SYSTEM_STATUS_COLS:
            log.info(f"[ast] Dropping spurious status filter on column '{col.name}'")
            dropped = True
            continue
        kept.append(pred)

    if not dropped:
        return False

    if not kept:
        expr.args.pop("where", None)
    else:
        combined = kept[0]
        for cond in kept[1:]:
            combined = sqlglot.exp.and_(combined, cond)
        where.set("this", combined)
    return True

def _remove_hallucinated_dates_in_ast(
    expr: "sqlglot.exp.Expression", base_table: str
) -> Tuple["sqlglot.exp.Expression", bool]:
    """
    Runs strip_llm_date_filters() (an external, string-based helper) from
    the same merged main pass as _strip_status_filters_in_ast(), right
    after it, on the same expr — rather than as its own separate pipeline
    step with a fresh parse/serialize round trip of its own sitting
    between the two. strip_llm_date_filters() still works on SQL text
    internally, so this serializes the (already status-filter-mutated)
    expr once, runs it, and reparses only if it actually changed
    anything. Returns (expr_to_use, changed) — the same expr object and
    changed=False if nothing was stripped, so the caller can skip
    re-serializing the query when this pass was a no-op.
    """
    sql = expr.sql(dialect="tsql")
    new_sql = strip_llm_date_filters(sql, _get_all_date_cols(base_table))
    if new_sql == sql:
        return expr, False
    try:
        return sqlglot.parse_one(new_sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE), True
    except Exception:
        return expr, False



_STMT_SPLIT_RE = re.compile(r"'(?:[^']|'')*'|(;)")

def _sqlglot_ensure_single_statement(sql: str) -> str:
    """Back-compat name — now delegates to the regex-based implementation."""
    return _ensure_single_statement_regex(sql)

def _ensure_single_statement_regex(sql: str) -> str:
    """
    Regex/scan-based replacement for the old sqlglot-based single-statement
    enforcement. Truncating to the first top-level SELECT doesn't need a
    full parse — we only need to find the first statement-terminating
    semicolon that isn't sitting inside a quoted string literal, and check
    whether there's another SELECT after it.

    The regex alternates between matching whole quoted string literals
    (so semicolons inside them are skipped over) and bare top-level
    semicolons; only the second alternative captures group 1.
    """
    if not sql or ";" not in sql:
        return sql
    for m in _STMT_SPLIT_RE.finditer(sql):
        if m.group(1) == ";":
            head = sql[: m.start()].strip()
            tail = sql[m.end():].strip()
            if head and tail and re.search(r"\bSELECT\b", tail, re.I):
                log.warning("[regex] Multiple top-level statements — truncating to first")
                return head
    return sql

def _enforce_join_clause(sql: str, verified_join_clause: str) -> str:
    """
    Swap the LLM's FROM/JOIN for the verified, graph-built one — but keep
    whatever table aliases the original query was using.

    Previously this dropped the original FROM/JOIN (aliases and all) and
    spliced in `verified_join_clause`, which is always built from real,
    unaliased table names (see build_join_clause / _find_direct_fk_join).
    If the LLM's SELECT/WHERE/ORDER BY referenced a table via an alias
    (e.g. `FROM MFC_M_ITEM AS T1 ... SELECT T1.Item_Name`), that alias
    simply vanished from the rebuilt FROM clause, leaving `T1.Item_Name`
    dangling — a "multipart identifier could not be bound" error, even
    though the base query logic was fine. Re-attaching the original
    aliases to the verified join's tables (and requalifying that join's
    own ON-clause columns to match) keeps the two halves consistent.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql")
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql
        where_node = expr.args.get("where")
        group_node = expr.args.get("group")
        having_node = expr.args.get("having")
        order_node = expr.args.get("order")
        limit_node = expr.args.get("limit")
        top_node = select.args.get("top")
        orig_alias_map: Dict[str, str] = {}
        for tnode in expr.find_all(sqlglot.exp.Table):
            real = (tnode.name or "").upper()
            alias = tnode.alias_or_name or ""
            if real and alias and alias.lower() != real.lower():
                orig_alias_map[real] = alias

        final_join_clause = verified_join_clause
        _real_to_alias: Dict[str, str] = {}
        if orig_alias_map:
            try:
                _jc_expr = sqlglot.parse_one(
                    f"SELECT 1 FROM {verified_join_clause}",
                    dialect="tsql",
                    error_level=_sqlglot_errors.ErrorLevel.IGNORE,
                )
                for jt in _jc_expr.find_all(sqlglot.exp.Table):
                    real = (jt.name or "").upper()
                    alias = orig_alias_map.get(real)
                    if alias:
                        jt.set("alias", sqlglot.exp.TableAlias(this=sqlglot.exp.to_identifier(alias)))
                        _real_to_alias[real] = alias
                if _real_to_alias:
                    for jcol in _jc_expr.find_all(sqlglot.exp.Column):
                        if jcol.table and jcol.table.upper() in _real_to_alias:
                            jcol.set(
                                "table",
                                sqlglot.exp.Identifier(this=_real_to_alias[jcol.table.upper()]),
                            )
                    _from = _jc_expr.find(sqlglot.exp.From)
                    if _from:
                        parts = [_from.this.sql(dialect="tsql")]
                        for j in _jc_expr.find_all(sqlglot.exp.Join):
                            parts.append(j.sql(dialect="tsql"))
                        final_join_clause = " ".join(parts)
                        log.info(
                            "[enforce_join] Preserved original alias(es) %s in rebuilt FROM/JOIN",
                            _real_to_alias,
                        )
            except Exception as _e:
                log.warning(
                    "[enforce_join] Could not preserve aliases (%s) — using unaliased join clause", _e
                )
                final_join_clause = verified_join_clause
                _real_to_alias = {}

        if _real_to_alias:
            
            for _col in expr.find_all(sqlglot.exp.Column):
                if _col.table and _col.table.upper() in _real_to_alias:
                    _col.set(
                        "table",
                        sqlglot.exp.Identifier(this=_real_to_alias[_col.table.upper()]),
                    )

        select_cols = ", ".join(e.sql(dialect="tsql") for e in select.expressions)
        top_sql = f"TOP {top_node.sql(dialect='tsql')} " if top_node else ""
        select_sql = f"SELECT {top_sql}{select_cols}"

        tail_parts = []
        if where_node:
            tail_parts.append(where_node.sql(dialect="tsql"))
        if group_node:
            tail_parts.append(group_node.sql(dialect="tsql"))
        if having_node:
            tail_parts.append(having_node.sql(dialect="tsql"))
        if order_node:
            tail_parts.append(order_node.sql(dialect="tsql"))
        if limit_node:
            tail_parts.append(limit_node.sql(dialect="tsql"))
        tail = " ".join(tail_parts)

        rebuilt = f"{select_sql} FROM {final_join_clause}"
        if tail:
            rebuilt = f"{rebuilt} {tail}"
        rebuilt = rebuilt.rstrip()
        if not rebuilt.endswith(";"):
            rebuilt += ";"
        return rebuilt
    except Exception as exc:
        log.warning(f"[enforce_join] sqlglot failed: {exc} — returning original SQL")
        return sql

_HAS_SQLGLOT_TOP = hasattr(sqlglot.exp, "Top")

def _enforce_top_limit_and_order_in_ast(
    expr: "sqlglot.exp.Expression",
    all_tables: List[str],
    full_schema: Optional[dict] = None,
    default_top: int = _DEFAULT_TOP,
) -> Tuple[bool, bool]:
    """
    Single-parse replacement for calling enforce_top_limit() then
    enforce_default_order() back to back. Both functions independently
    re-derived the same "is this an aggregate / GROUP BY query" facts from
    a freshly re-parsed AST; here those facts are computed once and both
    the TOP clause and ORDER BY clause are applied directly to the caller's
    already-parsed `expr` in place.

    Returns (changed, needs_top_text_splice):
      - changed: True if the AST was mutated (ORDER BY, and TOP when the
        installed sqlglot version exposes exp.Top).
      - needs_top_text_splice: True if this SQL needs a TOP clause but the
        installed sqlglot version has no exp.Top node to build one — some
        environments run older sqlglot builds without it. The caller must
        serialize `expr` and splice "TOP N" into the text itself (same
        fallback the original enforce_top_limit() used).
    """
    full_schema = FULL_SCHEMA_CACHE if full_schema is None else full_schema
    select = expr.find(sqlglot.exp.Select)
    if not select:
        return False, False

    _agg_types = (
        sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
        sqlglot.exp.Min, sqlglot.exp.Max,
    )
    has_group = expr.find(sqlglot.exp.Group) is not None
    has_agg = any(expr.find(t) is not None for t in _agg_types)
    skip_common = has_group or has_agg

    changed = False
    needs_top_text_splice = False

    # ── TOP ─────────────────────────────────────────────────────────────
    if not skip_common and not (select.args.get("top") or select.args.get("limit")):
        if _HAS_SQLGLOT_TOP:
            top_val = sqlglot.exp.Literal.number(default_top)
            select.set("top", sqlglot.exp.Top(this=top_val))
            changed = True
        else:
            needs_top_text_splice = True

    # ── ORDER BY ────────────────────────────────────────────────────────
    if not skip_common and not expr.find(sqlglot.exp.Order):
        _order_ctx: Optional["_SqlResolutionContext"] = None
        for table in all_tables:
            pks = full_schema.get(table, {}).get("pk", [])
            date_cols = full_schema.get(table, {}).get("date_columns", [])
            order_col = None
            if pks:
                order_col = f"{table}.{pks[0]}"
            elif date_cols:
                order_col = f"{table}.{date_cols[0]}"
            if order_col:
                if _order_ctx is None:
                   
                    _order_ctx = _SqlResolutionContext(expr, all_tables, scope_select=select)
                order_qualifier = _order_ctx.qualifier_for(table.upper())
                order_node = sqlglot.exp.Order(
                    expressions=[
                        sqlglot.exp.Ordered(
                            this=sqlglot.exp.column(order_col.split(".")[-1], table=order_qualifier),
                            desc=True,
                        )
                    ]
                )
                expr.set("order", order_node)
                changed = True
                break

    return changed, needs_top_text_splice

def _splice_top_into_sql(sql: str, default_top: int = _DEFAULT_TOP) -> str:
    """
    Text-level TOP injection for sqlglot builds that don't expose
    exp.Top — same fallback the original enforce_top_limit() used.
    """
    raw = sql.strip()
    if raw.upper().startswith("SELECT DISTINCT "):
        return raw[:16] + f"TOP {default_top} " + raw[16:]
    if raw.upper().startswith("SELECT "):
        return raw[:7] + f"TOP {default_top} " + raw[7:]
    return raw

def extract_first_valid_sql(text: str) -> Optional[str]:
    if not text:
        return None
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    try:
        statements = list(sqlglot.parse(text, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE))
        for stmt in statements:
            if stmt and stmt.key == "select":
                return stmt.sql(dialect="tsql")
    except Exception:
        pass
    match = re.search(r"\bSELECT\b", text, re.I)
    if not match:
        return None
    start = match.start()
    semicolon_pos = text.find(';', start)
    if semicolon_pos != -1:
        candidate = text[start:semicolon_pos+1].strip()
    else:
        rest = text[start:]
        next_select = None
        for m in re.finditer(r"\bSELECT\b", rest, re.I):
            if m.start() == 0:
                continue
            if start + m.start() > 0 and text[start + m.start() - 1] == '(':
                continue
            next_select = m.start()
            break
        candidate = rest[:next_select].strip() if next_select else rest.strip()
    if not re.match(r"\s*SELECT\b", candidate, re.I):
        candidate = "SELECT " + candidate
    return convert_mysql_to_sqlserver(candidate) if candidate else None

_RESULT_TABLE_INTENT_PATTERNS: List[str] = [
    r"\bwhich\b", r"\bwhat\b", r"\bshow\b", r"\blist\b", r"\bdetails?\b",
    r"\bassigned to\b", r"\bassociated with\b", r"\blinked to\b",
    r"\bbelongs? to\b", r"\bname of\b", r"\bgive me\b",
]
_RESULT_TABLE_INTENT_RE = re.compile("|".join(_RESULT_TABLE_INTENT_PATTERNS), re.IGNORECASE)


def _question_requests_table_data(question: str) -> bool:
    """
    True if the question's phrasing ("which routing is assigned to this
    item", "show the details of...", "what X is linked to...") indicates
    the user wants a related object's data returned — not just a yes/no
    existence filter. Used to protect a joined table from being demoted
    to a filter-only EXISTS semi-join in _resolve_join_shape() when the
    model's SELECT list simply forgot to include that table's columns.
    """
    if not question:
        return False
    return bool(_RESULT_TABLE_INTENT_RE.search(question))


def _pick_descriptive_columns(table: str, live_cols: Dict[str, List[str]], limit: int = 2) -> List[str]:
    """
    Pick up to `limit` non-PK, business-descriptive columns for `table`
    (role == "name": name/code/title/label/description/designation) from
    the columns that were actually sent to the LLM for this table. Falls
    back to any non-PK column if no "name"-role column is found, so a
    protected result table always gets *something* meaningful injected
    rather than nothing.
    """
    cols = live_cols.get(table, []) or live_cols.get(table.upper(), [])
    if not cols:
        return []
    pk_cols = _get_table_pk_cols(table)
    roles = COL_ROLE_INDEX.get(table, {}) or COL_ROLE_INDEX.get(table.upper(), {})

    descriptive = [
        c for c in cols
        if c.lower() not in pk_cols and "name" in (roles.get(c, roles.get(c.lower(), [])))
    ]
    if descriptive:
        return descriptive[:limit]

    non_pk = [c for c in cols if c.lower() not in pk_cols]
    return non_pk[:limit] if non_pk else cols[:limit]


def _resolve_join_shape(
    sql: str,
    base_table: str,
    intent: str,
    question: str = "",
    live_cols: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Rewrites JOINs to N-side (transaction) tables that are used ONLY as a
    filter — never actually displayed, grouped, or ordered by — into
    correlated EXISTS semi-joins, so they can never fan out the base
    table's rows into duplicates (the classic 1:N JOIN duplication
    problem). Real JOINs to tables whose columns the query actually
    selects/groups/orders by are always left untouched, since row
    multiplication there reflects real data, not a bug — dropping or
    collapsing those rows would silently discard information the user
    asked for.

    Derives everything from the SQL's OWN real FROM/JOIN via sqlglot,
    never from caller bookkeeping (join_tables/join_clause), matching the
    same "trust the AST, not the bookkeeping" principle used throughout
    this module (see _SqlResolutionContext). 1-side vs N-side is decided
    generically via _is_master_table()'s TABLE_NAME_PREFIXES convention —
    no per-table or per-module rules required, so this applies uniformly
    across every module (MFC, USR, and any future module) without
    maintenance.

    Skipped entirely for aggregation-style intents, where GROUP BY already
    compacts N-side rows correctly — a real JOIN there is the intended
    shape, not the bug this function targets.

    Result-table protection: before demoting a table with no SELECT/GROUP/
    ORDER reference, check whether the user's question phrasing indicates
    they actually want that related object's data (_question_requests_
    table_data). If so, this is very likely the model forgetting to select
    the table's columns — not a genuine filter-only usage — so instead of
    converting to EXISTS, inject that table's descriptive column(s) into
    SELECT and keep the real JOIN. Only falls through to EXISTS conversion
    if injection isn't possible (no live_cols available for that table).
    """
    if not sql or intent in ("relational_aggregation", "aggregation", "group_by"):
        return sql

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if select is None:
            return sql

        working_joins: List = list(select.args.get("joins") or [])
        if not working_joins:
            return sql

        base_upper = base_table.upper()
        requested_keys: set = set()
        for scope_key in ("expressions", "group", "order"):
            for node in select.args.get(scope_key) or []:
                for col in node.find_all(sqlglot.exp.Column):
                    if col.table:
                        requested_keys.add(col.table.upper())

        changed = False
        for join in list(working_joins):
            tbl_node = join.this
            if not isinstance(tbl_node, sqlglot.exp.Table):
                continue
            tname = (tbl_node.name or "").upper()
            ref_key = (tbl_node.alias_or_name or tname).upper()
            if not tname or tname == base_upper:
                continue
            if _is_master_table(tname):
                continue
         
            if ref_key in requested_keys or tname in requested_keys:
                continue

      
            if live_cols and _question_requests_table_data(question):
                real_table_name = tbl_node.name
                pick_cols = _pick_descriptive_columns(real_table_name, live_cols)
                if pick_cols:
                    qualifier = tbl_node.alias_or_name or real_table_name
                    for col_name in pick_cols:
                        select.append(
                            "expressions",
                            sqlglot.exp.Column(
                                this=sqlglot.exp.Identifier(this=col_name),
                                table=sqlglot.exp.Identifier(this=qualifier),
                            ),
                        )
                    changed = True
                    log.info(
                        "[join_shape] Protected result table '%s' from EXISTS demotion — "
                        "question implies user wants its data; injected column(s): %s",
                        tname, pick_cols,
                    )
                    _vprint(
                        "PIPELINE", "green",
                        f"🛡️ Kept JOIN for {tname}: question asks for its data — "
                        f"injected {', '.join(pick_cols)} instead of collapsing to EXISTS",
                    )
                    continue

            on_expr = join.args.get("on")
            if on_expr is None:
                continue
            is_bridge = False
            for other in working_joins:
                if other is join:
                    continue
                other_on = other.args.get("on")
                if other_on is None:
                    continue
                if any(
                    (c.table or "").upper() == ref_key
                    for c in other_on.find_all(sqlglot.exp.Column)
                ):
                    is_bridge = True
                    break
            if is_bridge:
                continue
            where_node = select.args.get("where")
            moved: List = []
            if where_node is not None and where_node.this is not None:
                conjuncts = (
                    list(where_node.this.flatten())
                    if isinstance(where_node.this, sqlglot.exp.And)
                    else [where_node.this]
                )
                keep = []
                for c in conjuncts:
                    refs_table = any(
                        (col.table or "").upper() == ref_key
                        for col in c.find_all(sqlglot.exp.Column)
                    )
                    (moved if refs_table else keep).append(c)
                if keep:
                    new_where = keep[0]
                    for c in keep[1:]:
                        new_where = sqlglot.exp.And(this=new_where, expression=c)
                    select.set("where", sqlglot.exp.Where(this=new_where))
                else:
                    select.set("where", None)

            sub_where = on_expr.copy()
            for c in moved:
                sub_where = sqlglot.exp.And(this=sub_where, expression=c.copy())

            subquery = (
                sqlglot.exp.Select(expressions=[sqlglot.exp.Literal.number(1)])
                .from_(tbl_node.copy())
                .where(sub_where)
            )
            exists_expr = sqlglot.exp.Exists(this=subquery)

            cur_where = select.args.get("where")
            if cur_where is not None and cur_where.this is not None:
                select.set("where", sqlglot.exp.Where(
                    this=sqlglot.exp.And(this=cur_where.this, expression=exists_expr)
                ))
            else:
                select.set("where", sqlglot.exp.Where(this=exists_expr))

            working_joins.remove(join)
            select.set("joins", working_joins)
            changed = True

            log.info(
                "[join_shape] Converted N-side JOIN -> EXISTS semi-join: table=%s "
                "(no SELECT/GROUP/ORDER reference — filter-only usage, prevents row fan-out)",
                tname,
            )
            _vprint(
                "PIPELINE", "green",
                f"\U0001f500 JOIN \u2192 EXISTS: {tname} used only to filter, not displayed \u2014 row fan-out prevented",
            )

        if not changed:
            return sql
        return expr.sql(dialect="tsql")

    except Exception as exc:
        log.warning("[join_shape] Failed to resolve join shape (%s) — returning original SQL", exc)
        return sql


# ============================================================
# ===== DISTINCT IMPLEMENTATION =====
# ============================================================

def _has_window_functions(expr) -> bool:
    """
    Check if the SQL has window functions.
    Uses only window function types that exist in the current sqlglot version.
    """
    try:
        # Check for OVER clause (window functions use OVER)
        sql = expr.sql(dialect="tsql")
        if re.search(r'\bOVER\s*\(', sql, re.IGNORECASE):
            return True
    except Exception:
        pass
    
    # Try to find Window class if it exists
    if hasattr(sqlglot.exp, 'Window'):
        try:
            if expr.find(sqlglot.exp.Window) is not None:
                return True
        except Exception:
            pass
    
    return False

def _is_one_to_many_join(
    left_table: str,
    right_table: str,
    left_alias: str,
    right_alias: str,
    on_expr,
    alias_map: Dict[str, str],
    schema: Dict[str, dict]
) -> bool:
    """
    Determine if a join is 1:N (one-to-many) based on PK coverage.
    
    Returns True if:
    - The left side's ON columns cover its primary key (parent/one side)
    - The right side's ON columns do NOT cover its primary key (child/many side)
    
    This is the most reliable way to detect 1:N joins without naming conventions.
    """
    def _get_pk_columns(table: str) -> set:
        """Get primary key columns for a table."""
        # Check FULL_SCHEMA_CACHE first
        pk = {c.lower() for c in schema.get(table, {}).get("pk", [])}
        if pk:
            return pk
        
        # Fallback to META
        for col, info in META.get(table, {}).get("columns", {}).items():
            if isinstance(info, dict) and info.get("is_pk", False):
                pk.add(col.lower())
        
        # Fallback to COL_ROLE_INDEX
        for col, roles in COL_ROLE_INDEX.get(table, {}).items():
            if "identifier" in roles:
                pk.add(col.lower())
        
        return pk
    
    def _get_on_columns(alias: str) -> set:
        """Get columns from the ON clause for a specific alias."""
        cols = set()
        for eq in on_expr.find_all(sqlglot.exp.EQ):
            for side in (eq.this, eq.expression):
                if isinstance(side, sqlglot.exp.Column):
                    if (side.table or "").upper() == alias.upper():
                        cols.add(side.name.lower())
        return cols
    left_pk = _get_pk_columns(left_table)
    right_pk = _get_pk_columns(right_table)
    left_on_cols = _get_on_columns(left_alias)
    right_on_cols = _get_on_columns(right_alias)
    left_is_one = bool(left_pk) and left_pk.issubset(left_on_cols)
    right_is_one = bool(right_pk) and right_pk.issubset(right_on_cols)
    if left_is_one and not right_is_one:
        return True
    if right_is_one and not left_is_one:
        return True
    if left_is_one and right_is_one:
        return False
    return False


def _should_apply_distinct(
    sql: str,
    intent: str,
    join_tables: List[str],
    full_schema: Optional[Dict[str, dict]] = None,
) -> bool:
    """
    Determine if DISTINCT should be applied to a join query.
    
    Returns True ONLY when ALL of these conditions are met:
    1. Multi-table join query
    2. Not an aggregation/group_by query (GROUP BY already handles duplication)
    3. At least one verified 1:N join exists
    4. Only columns from the "one" side (parent) are projected
    5. The query doesn't already have DISTINCT
    6. No window functions present
    
    This is intentionally conservative to avoid incorrect results.
    """
    # Quick checks before parsing
    if not sql or intent in ("relational_aggregation", "aggregation", "group_by"):
        return False
    
    # Check if query has multiple tables and JOINs
    if len(join_tables) < 2:
        return False
    
    if "JOIN" not in sql.upper():
        return False
    
    # Check if SELECT already has DISTINCT
    if re.search(r"\bSELECT\s+DISTINCT\b", sql, re.IGNORECASE):
        return False
    
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if select is None:
            return False
        
        # Skip if window functions are present (DISTINCT + window functions can be tricky)
        if _has_window_functions(expr):
            log.info("[distinct] Window functions present - skipping DISTINCT")
            return False
        
        joins = list(select.args.get("joins") or [])
        if not joins:
            return False
        
        # Build alias -> real table name mapping
        alias_map: Dict[str, str] = {}
        from_ = select.find(sqlglot.exp.From)
        if from_ is not None and isinstance(from_.this, sqlglot.exp.Table):
            t = from_.this
            alias_map[(t.alias_or_name or t.name).upper()] = (t.name or "").upper()
        for j in joins:
            if isinstance(j.this, sqlglot.exp.Table):
                t = j.this
                alias_map[(t.alias_or_name or t.name).upper()] = (t.name or "").upper()
        
        # Find which tables are projected (appear in SELECT, GROUP BY, ORDER BY)
        projected_tables: set = set()
        for scope in ("expressions", "group", "order"):
            for node in select.args.get(scope) or []:
                for col in node.find_all(sqlglot.exp.Column):
                    if col.table:
                        projected_tables.add(col.table.upper())
        
        # If no specific tables are projected (SELECT *), be extra conservative
        if not projected_tables:
            # Check if SELECT * is used
            for exp in select.expressions:
                if isinstance(exp, sqlglot.exp.Star):
                    log.info("[distinct] SELECT * detected - skipping DISTINCT (too risky)")
                    return False
        
        # Analyze each join for 1:N cardinality
        many_side_tables: set = set()
        schema = full_schema if full_schema is not None else FULL_SCHEMA_CACHE
        
        for join in joins:
            if not isinstance(join.this, sqlglot.exp.Table):
                continue
            
            right_alias = (join.this.alias_or_name or join.this.name).upper()
            on_expr = join.args.get("on")
            if on_expr is None:
                continue
            
            # Find the left side of the join
            on_tables = {c.table.upper() for c in on_expr.find_all(sqlglot.exp.Column) if c.table}
            left_alias = next((t for t in on_tables if t != right_alias and t in alias_map), None)
            if left_alias is None:
                continue
            
            # Get real table names
            left_table = alias_map.get(left_alias, left_alias)
            right_table = alias_map.get(right_alias, right_alias)
            
            # Check cardinality
            is_one_to_many = _is_one_to_many_join(
                left_table=left_table,
                right_table=right_table,
                left_alias=left_alias,
                right_alias=right_alias,
                on_expr=on_expr,
                alias_map=alias_map,
                schema=schema
            )
            
            if is_one_to_many:
                # This is a 1:N join - the right side is the "many" side
                many_side_tables.add(right_alias)
                log.info("[distinct] Found 1:N join: %s -> %s", left_alias, right_alias)
        
        if not many_side_tables:
            log.info("[distinct] No 1:N joins found")
            return False
        
        # Check if any many-side table is projected
        projected_many_side = projected_tables & many_side_tables
        if projected_many_side:
            log.info("[distinct] Many-side columns projected (%s) - skipping DISTINCT", projected_many_side)
            return False
        
        log.info("[distinct] DISTINCT should be applied (1:N joins with only parent-side projection)")
        return True
        
    except Exception as exc:
        log.warning("[distinct] Error analyzing query: %s - skipping DISTINCT", exc)
        return False

def _fix_distinct_order_by(sql: str) -> str:
    """
    Fix ORDER BY for SELECT DISTINCT queries.
    Ensures all ORDER BY columns appear in SELECT list.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql
        
        # Check if SELECT DISTINCT is present
        if not select.args.get("distinct"):
            return sql
        
        order = expr.find(sqlglot.exp.Order)
        if not order:
            return sql

        select_cols = set()
        select_aliases = set()

        for proj in select.expressions:
            target = proj.this if isinstance(proj, sqlglot.exp.Alias) else proj
            if isinstance(target, sqlglot.exp.Column):
                if target.table and target.name:
                    select_cols.add(f"{target.table.lower()}.{target.name.lower()}")
                elif target.name:
                    select_cols.add(target.name.lower())
            if isinstance(proj, sqlglot.exp.Alias):
                select_aliases.add(proj.alias.lower())
        
        # Check each ORDER BY item
        order_cols_to_add = []
        
        for order_item in order.find_all(sqlglot.exp.Ordered):
            inner = order_item.this
            
            # Check if it's a Column
            if isinstance(inner, sqlglot.exp.Column):
                col_key = None
                if inner.table and inner.name:
                    col_key = f"{inner.table.lower()}.{inner.name.lower()}"
                elif inner.name:
                    col_key = inner.name.lower()
                
                # If column not in SELECT and not an alias, add it
                if col_key and col_key not in select_cols and inner.name.lower() not in select_aliases:
                    new_col = inner.copy()
                    select.append("expressions", new_col)
                    order_cols_to_add.append(inner.name)
        
        if order_cols_to_add:
            log.info("[fix_distinct_order] Added columns to SELECT for ORDER BY: %s", order_cols_to_add)
            _vprint(
                "PIPELINE", "yellow",
                f" Added ORDER BY columns to SELECT: {order_cols_to_add}",
                "Required for SELECT DISTINCT"
            )
        
        return expr.sql(dialect="tsql")
    
    except Exception as exc:
        log.warning("[fix_distinct_order] Failed: %s", exc)
        return sql

# ============================================================
# ===== END DISTINCT IMPLEMENTATION =====
# ============================================================


def post_process_sql(
    sql: str,
    base_table: str,
    question: str,
    intent: str,
    join_tables: List[str],
    join_clause: Optional[str] = None,
    skip_lov_fix: bool = False,
    shared_cols: Optional[Dict[str, List[str]]] = None,
    live_cols: Optional[Dict[str, List[str]]] = None,
) -> str:
    _PARAM_SENTINEL = "__PARAM_PLACEHOLDER__"
    _param_count = sql.count("?")
    if _param_count:
        sql = sql.replace("?", f"'{_PARAM_SENTINEL}'")

    all_tables = list(join_tables) if join_tables else [base_table]

    if join_clause:
        try:
            _jc_expr = sqlglot.parse_one(
                f"SELECT 1 FROM {join_clause}",
                dialect="tsql",
                error_level=_sqlglot_errors.ErrorLevel.IGNORE,
            )
            for _join in _jc_expr.find_all(sqlglot.exp.Join):
                _tname = (_join.this.name or "").upper()
                if _tname and _tname not in {t.upper() for t in all_tables}:
                    all_tables.append(_tname)
                    log.info("[post_process] Added intermediate join table '%s' to all_tables", _tname)
        except Exception as _e:
            log.debug("[post_process] Could not parse join_clause for intermediate tables: %s", _e)

    sql = scrub_invalid_sql_tokens(sql)

    if not sql.upper().strip().startswith("SELECT"):
        log.error("[post_process] SQL does not start with SELECT: %s", sql[:200])
        return ""

    if sql.upper().count("SELECT") > 1:
        extracted = extract_first_valid_sql(sql)
        if extracted:
            sql = extracted

    if " FROM " not in sql.upper():
        from_clause = join_clause if (join_clause and len(join_tables) > 1) else base_table
        sql = sql.rstrip().rstrip(";") + f" FROM {from_clause}"
        log.warning("[post_process] FROM clause missing — recovered with: FROM %s", from_clause)
    if join_clause and len(join_tables) > 1:
        sql = _enforce_join_clause(sql, join_clause)
    sql = _resolve_columns(sql, all_tables, live_cols=live_cols)
    sql = _ensure_single_statement_regex(sql)
    try:
        _expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        _status_changed = _strip_status_filters_in_ast(_expr, question)
        _expr, _date_changed = _remove_hallucinated_dates_in_ast(_expr, base_table)
        if _status_changed or _date_changed:
            sql = _expr.sql(dialect="tsql")
    except Exception as _e:
        log.warning("[post_process] status/date AST pass failed (%s) — falling back to string passes", _e)
        sql = _remove_hallucinated_date_filters(sql, base_table)

    if not skip_lov_fix:
        sql = _fix_lov_values_consolidated(sql, all_tables)
    sql = _fix_all_string_literals(sql, all_tables)

    # ── Date/time filter injection ─────────────────────────────────────────────
    if _dr_has_temporal_intent(question):
        sql = inject_time_filter(sql, base_table, question)
    if _dr_has_effective_intent(question):
        sql = inject_effective_filter(sql, base_table, question)

    if intent not in _NO_TOP_INTENTS:
        try:
            _expr2 = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
            _changed, _needs_top_splice = _enforce_top_limit_and_order_in_ast(
                _expr2, all_tables, FULL_SCHEMA_CACHE
            )
            if _changed:
                sql = _expr2.sql(dialect="tsql")
            if _needs_top_splice:
                sql = _splice_top_into_sql(sql)
        except Exception as _e:
            log.warning("[post_process] TOP/ORDER BY AST pass failed (%s) — sql left unchanged", _e)
    sql = _sqlglot_replace_select_star(sql, all_tables, question, intent=intent)
    sql = _fix_group_by_coherence(sql)
    sql = _clean_order_and_group_by(sql, all_tables, live_cols=live_cols)
    sql = _resolve_columns(sql, all_tables, live_cols=live_cols)

    # ── 1:N join-shape resolution ──────────────────────────────────────────
    if len(all_tables) > 1:
        _pre_shape_sql = sql
        sql = _resolve_join_shape(sql, base_table, intent, question=question, live_cols=live_cols)
        if sql != _pre_shape_sql:
       
            sql = _resolve_columns(sql, all_tables, live_cols=live_cols)

        # ── Conservative DISTINCT injection ──────────────────────────────────
        if _should_apply_distinct(sql, intent, join_tables, FULL_SCHEMA_CACHE):
            if not re.search(r"\bSELECT\s+DISTINCT\b", sql, re.IGNORECASE):
                # ── Add DISTINCT first ───────────────────────────────────────────
                sql = re.sub(r"\bSELECT\b", "SELECT DISTINCT", sql, count=1, flags=re.IGNORECASE)
                log.info("[post_process] Added DISTINCT (safe 1:N join with parent-only projection)")
                _vprint(
                    "PIPELINE", "green",
                    " Added DISTINCT - safe 1:N join with parent-only projection",
                    f"tables={join_tables}"
                )

                # ── FIX: Ensure ORDER BY columns are in SELECT ───────────────────
                try:
                    test_expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
                    if test_expr.find(sqlglot.exp.Order) is not None:
                        sql = _fix_distinct_order_by(sql)
                except Exception:
                    pass

        try:
            _final = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
            _sel = _final.find(sqlglot.exp.Select)
            _frm = _final.find(sqlglot.exp.From)
            if _sel is not None and not _sel.expressions and _frm is not None:
                log.error("[post_process] FINAL SQL has empty column list — aborting")
                return ""
            sql = _final.sql(dialect="tsql")
        except Exception:
            pass

    if _param_count:
        sql = re.sub(r"N?'__PARAM_PLACEHOLDER__'", "?", sql)

    return sql.strip()


def _col_workflow_score(col: str, question: str) -> float:
    cl, ql, score = col.lower(), question.lower(), 0.0
    if any(cl.endswith(sfx) for sfx in _COL_WORKFLOW_LOV_SUFFIXES):
        score += 3.0
    elif any(cl.endswith(sfx) for sfx in _COL_WORKFLOW_TYPE_SUFFIXES):
        score += 1.0
    if cl in _SYSTEM_STATUS_COLS:
        score -= 2.5
    col_words = cl.replace("_", " ")
    if col_words in ql:
        score += 2.0
    else:
        for word in col_words.split():
            if len(word) > 3 and word in ql:
                score += 0.8
                break
    return score

def resolve_lov_matches(
    question: str,
    table: str,
    _precomputed_fuzzy: Optional[Dict[str, Optional[str]]] = None,
) -> dict:
    ql = question.lower()
    lov = get_lov_for_table(table)
    raw: Dict[str, List[Tuple]] = {}
    _fuzzy_map: Optional[Dict[str, Optional[str]]] = _precomputed_fuzzy

    if lov:
        for col, mapping in lov.items():
            if col.lower() in _SYSTEM_STATUS_COLS and not _question_implies_system_status(question):
                continue
            for _raw_key, _meta in mapping.items():
                label, db_val = _lov_label_value(_raw_key, _meta)
                ll = label.lower()
                matched_phrase: Optional[str] = None

                if ll in ql:
                    matched_phrase = ll
                else:
                    canonical = _canonical_label(ll)
                    if canonical:
                        best: Optional[str] = None
                        for phrase, canon in EmbeddingStore.instance().get_phrase_to_label().items():
                            if canon == canonical and phrase in ql:
                                if best is None or len(phrase) > len(best):
                                    best = phrase
                        if best:
                            matched_phrase = best

                    if matched_phrase is None and EmbeddingStore.instance().lov_embed_matrix is not None:
                        candidates = [w for w in re.findall(r'[a-z]{3,}', ql)]

                        for cand in candidates:
                            if len(cand) < 3:
                                continue
                            if _fuzzy_map is not None:
                                fuzzy_lbl = _fuzzy_map.get(cand)
                            else:
                                fuzzy_lbl = _fuzzy_lov_match(cand, threshold=_LOV_FUZZY_THRESHOLD)
                            if fuzzy_lbl and fuzzy_lbl == ll:
                                matched_phrase = cand
                                log.info(f"[LOV:fuzzy] '{cand}' → '{ll}' (embedding match)")
                                break

                if matched_phrase:
                    raw.setdefault(col, []).append((label, db_val, matched_phrase))

    _question_has_literal = bool(extract_literal_value(question))
    type_filters = (
        [] if _question_has_literal
        else get_type_filters_all(table, question, min_score=0.6)
    )
    for tf in type_filters:
        col = tf["col"]
        db_val = tf["value"]
        label = tf["label"]
        if col not in raw:
            raw.setdefault(col, []).append((label, db_val, label.lower()))

    if not raw:
        return {}

    deduped: Dict[str, List[Tuple]] = {}
    for col, matches in raw.items():
        seen_vals: set = set()
        deduped[col] = []
        for label, db_val, phrase in sorted(matches, key=lambda x: len(x[2]), reverse=True):
            if db_val not in seen_vals:
                deduped[col].append((label, db_val, phrase))
                seen_vals.add(db_val)

    non_sys = {c: v for c, v in deduped.items() if c.lower() not in _SYSTEM_STATUS_COLS}
    sys_only = {c: v for c, v in deduped.items() if c.lower() in _SYSTEM_STATUS_COLS}
    candidates = non_sys if non_sys else sys_only

    result = {}
    for col, matches in sorted(
        candidates.items(),
        key=lambda kv: _col_workflow_score(kv[0], question),
        reverse=True,
    ):
        if len(matches) == 1:
            label, db_val, _ = matches[0]
            result[col] = db_val
            log.info(f"[LOV] table={table} col={col} label='{label}' → {db_val}")
        else:
            vals = [db_val for _, db_val, _ in matches]
            labels = [label for label, _, _ in matches]
            result[col] = vals
            log.info(f"[LOV] table={table} col={col} multi={labels} → IN {vals}")
    return result

def build_lov_context_block(table: str) -> str:
    return EmbeddingStore.instance().get_lov_context(table)

def fix_lov_in_sql(sql: str, table: str) -> str:
    """Back-compat single-table wrapper around the consolidated LOV pass."""
    return _fix_lov_values_consolidated(sql, [table])

def _fix_lov_values_consolidated(sql: str, all_tables: List[str]) -> str:
    """
    Single-parse replacement for calling fix_lov_in_sql() once per table.

    Previously post_process_sql looped `for t in all_tables: sql =
    fix_lov_in_sql(sql, t)`, which re-parsed and re-serialized the whole
    query with sqlglot once per table (N tables -> N full parse/print
    cycles). Here we build one merged LOV index across every table up
    front and walk the AST exactly once. Later tables win ties on
    (col, label) keys, matching the old sequential-overwrite behavior.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)
    except Exception:
        return sql

    lov_index: Dict[Tuple[str, str], str] = {}
    lov_abbrev_index: Dict[Tuple[str, str], str] = {}
    known_cols: set = set()

    for table in all_tables:
        known_cols.update(c.lower() for c in META.get(table, {}).get("columns", {}).keys())

        lov = get_lov_for_table(table)
        if not lov:
            continue
        for col, mapping in lov.items():
            cl = col.lower()
            for _raw_key, _meta in mapping.items():
                label, db_val = _lov_label_value(_raw_key, _meta)
                lov_index[(cl, label.lower())] = db_val
                lov_index.setdefault((cl, str(_raw_key).lower()), str(_raw_key))

            first_char_to_vals: Dict[str, List[str]] = {}
            for _raw_key, _meta in mapping.items():
                label, db_val = _lov_label_value(_raw_key, _meta)
                fc = label.lower()[0] if label else ""
                first_char_to_vals.setdefault(fc, []).append(db_val)
            for fc, vals in first_char_to_vals.items():
                if len(vals) == 1:
                    lov_abbrev_index[(cl, fc)] = vals[0]

    changed = False
    for eq in expr.find_all(sqlglot.exp.EQ):
        left, right = eq.left, eq.right
        if not isinstance(left, sqlglot.exp.Column):
            continue
        if not isinstance(right, sqlglot.exp.Literal):
            continue
        col_name = left.name.lower()
        label_val = right.this.lower()

        db_val = lov_index.get((col_name, label_val))

        if db_val is None and len(label_val) == 1:
            db_val = lov_abbrev_index.get((col_name, label_val))
            if db_val is not None:
                log.info(
                    "[_fix_lov_values_consolidated] Abbrev match: col=%r literal=%r -> db_val=%r",
                    col_name, label_val, db_val,
                )

        if db_val is not None:
            if db_val.lstrip("-").isdigit():
                eq.set("right", sqlglot.exp.Literal.number(db_val))
            else:
                eq.set("right", sqlglot.exp.Literal.string(db_val))
            changed = True

    for ident in list(expr.find_all(sqlglot.exp.Identifier)):
        if ident.quoted and ident.name.lower() not in known_cols:
            parent = ident.parent
            if isinstance(parent, sqlglot.exp.Column):
                continue
            if parent is not None:
                parent.replace(sqlglot.exp.null())
                changed = True

    if not changed:
        return sql
    try:
        return expr.sql(dialect="tsql")
    except Exception:
        return sql

def _extract_question_words(question: str) -> List[str]:
    return [w for w in re.findall(r'[a-z]{3,}', question.lower())]

def _has_valid_join_path(tables: List[str]) -> Optional[str]:
    if len(tables) < 2:
        return None
    try:
        clause = build_join_clause(tables[0], tables, intent="simple")
        return clause if (clause and "JOIN" in clause.upper()) else None
    except Exception:
        return None

# ── Prompt schema compression ─────────────────────────────────────────────────

def build_query_aware_lov_context(table: str, question: str, prompt_cols: List[str]) -> str:
    full_ctx = build_lov_context_block(table)
    if not full_ctx or not prompt_cols:
        return full_ctx

    prompt_cols_lower = {c.lower() for c in prompt_cols}
    filtered_lines: List[str] = []
    header_pending: Optional[str] = None

    for line in full_ctx.splitlines():
        if not line.startswith(" "):
            header_pending = line
            continue
        col_name = line.strip().split(":")[0].strip().lower()
        if col_name in prompt_cols_lower:
            if header_pending is not None:
                filtered_lines.append(header_pending)
                header_pending = None
            filtered_lines.append(line)

    result = "\n".join(filtered_lines).strip()
    if not result:
        return ""

    kept = sum(1 for l in filtered_lines if l.startswith(" "))
    total = sum(1 for l in full_ctx.splitlines() if l.startswith(" "))
    log.info(
        "[prompt_compress] LOV compression: table=%r kept %d/%d mappings",
        table, kept, total,
    )
    return result

def _remove_hallucinated_date_filters(sql: str, table: str) -> str:
    return strip_llm_date_filters(sql, _get_all_date_cols(table))

def get_shared_columns(tables: List[str], live_cols: Dict[str, List[str]]) -> Dict[str, List[str]]:
    col_map: Dict[str, List[str]] = {}
    for table in tables:
        for col in live_cols.get(table, []):
            col_map.setdefault(col.lower(), []).append(table)
    return {col: tbls for col, tbls in col_map.items() if len(tbls) > 1}


def _ast_fix_column_errors(sql: str, schema_errors: List[str]) -> Optional[str]:
    """
    Attempt a fast, LLM-free fix for column validation errors by rewriting the
    SQL AST with sqlglot.

    Strategy
    --------
    1. Parse the error messages produced by ``validate_sql`` to collect a set of
       (table_upper, column_lower) pairs that are invalid.
    2. Parse the SQL with sqlglot (T-SQL dialect, errors ignored).
    3. Walk every ``sqlglot.exp.Column`` node; if it references a bad column on a
       known alias/table, remove it from the enclosing SELECT, WHERE, GROUP BY,
       ORDER BY, or HAVING clause.
    4. If removing a column would leave a clause completely empty (e.g. a lone
       bad column in a WHERE), drop the entire clause rather than emit broken SQL.
    5. Return the regenerated SQL string, or ``None`` if parsing fails or nothing
       could be changed (so the caller can fall through to the LLM fix path).

    Only ``Column 'X' does not exist in table 'Y'.`` errors are handled here;
    unknown-table errors are left to the LLM fix path.
    """
    if not schema_errors or not sql:
        return None

    # ── 1. Extract bad (table_upper, col_lower) pairs from error strings ──────
    _COL_ERR_RE = re.compile(
        r"Column '([^']+)' does not exist in table '([^']+)'", re.IGNORECASE
    )
    bad_cols: set = set()
    for err in schema_errors:
        m = _COL_ERR_RE.search(err)
        if m:
            bad_cols.add((m.group(2).upper(), m.group(1).lower()))

    if not bad_cols:
        return None

    try:
        tree = sqlglot.parse_one(sql, dialect="tsql",
                                 error_level=_sqlglot_errors.ErrorLevel.IGNORE)
    except Exception as exc:
        log.debug("[_ast_fix_column_errors] sqlglot parse failed: %s", exc)
        return None

    # ── 2. Build alias → real-table map ──────────────────────────────────────
    alias_map: Dict[str, str] = {}
    for tnode in tree.find_all(sqlglot.exp.Table):
        real = tnode.name.upper()
        alias = tnode.alias_or_name.upper()
        alias_map[real.lower()] = real
        if alias.lower() != real.lower():
            alias_map[alias.lower()] = real

    def _is_bad_col(col_node: sqlglot.exp.Column) -> bool:
        if not col_node.table:
            return False
        tbl = alias_map.get(col_node.table.lower())
        if tbl is None:
            return False
        return (tbl.upper(), col_node.name.lower()) in bad_cols

    # ── 3. Remove bad column nodes from their parent expressions ─────────────
    changed = False
    def _detach(node: sqlglot.exp.Expression) -> None:
        nonlocal changed
        parent = node.parent
        if parent is None:
            return
        for key, val in parent.args.items():
            if isinstance(val, list) and node in val:
                val.remove(node)
                changed = True
                return
        node.replace(sqlglot.exp.Null())
        changed = True
    cols_to_remove = [c for c in tree.find_all(sqlglot.exp.Column) if _is_bad_col(c)]

    for col in cols_to_remove:
        parent = col.parent
        if parent is None:
            continue
        if isinstance(parent, sqlglot.exp.Alias):
            grandparent = parent.parent
            if grandparent is not None:
                for key, val in grandparent.args.items():
                    if isinstance(val, list) and parent in val:
                        val.remove(parent)
                        changed = True
                        break
                else:
                    parent.replace(sqlglot.exp.Null())
                    changed = True
        else:
            _detach(col)

    if not changed:
        return None

    # ── 4. Drop clauses that are now empty ───────────────────────────────────
    for clause_key in ("group", "order", "having", "where"):
        clause = tree.args.get(clause_key)
        if clause is None:
            continue
        expressions = clause.args.get("expressions", [])
        if isinstance(expressions, list) and len(expressions) == 0:
            tree.set(clause_key, None)
        elif clause_key == "where":
            inner = clause.args.get("this")
            if inner is None or isinstance(inner, sqlglot.exp.Null):
                tree.set("where", None)

    # ── 5. Guard: SELECT must keep at least one real expression ──────────────
    select_exprs = tree.args.get("expressions", [])
    real_exprs = [
        e for e in select_exprs
        if not isinstance(e, sqlglot.exp.Null)
    ]
    if not real_exprs:
        log.debug("[_ast_fix_column_errors] All SELECT columns were bad — cannot fix without LLM")
        return None

    try:
        fixed_sql = tree.sql(dialect="tsql")
        log.info(
            "[_ast_fix_column_errors] Stripped %d bad column reference(s); fixed SQL: %.200s",
            len(cols_to_remove), fixed_sql,
        )
        return fixed_sql
    except Exception as exc:
        log.debug("[_ast_fix_column_errors] SQL generation failed after AST fix: %s", exc)
        return None


def validate_sql(sql: str, selected_tables: List[str]) -> List[str]:
    errors: List[str] = []
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return errors
    if not selected_tables:
        log.warning("[validate_sql] No selected_tables provided — skipping validation.")
        return errors

    log.info("[validate_sql] Validating SQL against selected tables: %s", selected_tables)

    # ── Strict syntax check ──────────────────────────────────────────────
    try:
        sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.RAISE)
    except Exception as exc:
        err = f"SQL syntax error: {exc}"
        errors.append(err)
        log.warning("[validate_sql] %s", err)
        log.warning("[validate_sql] %d validation error(s) for tables %s: %s", len(errors), selected_tables, errors)
        return errors

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        all_sql_tables = {
            (tnode.name or "").upper() for tnode in expr.find_all(sqlglot.exp.Table)
        }
        schema_snapshot: Dict[str, dict] = {}
        for tbl_upper in (all_sql_tables | {t.upper() for t in selected_tables}):
            if not tbl_upper:
                continue
            entry = FULL_SCHEMA_CACHE.get(tbl_upper, {}) or FULL_SCHEMA_CACHE.get(tbl_upper.upper(), {})
            if not entry:
                for k in FULL_SCHEMA_CACHE:
                    if k.upper() == tbl_upper:
                        entry = FULL_SCHEMA_CACHE[k]
                        break
            cols = entry.get("columns", [])
            if cols:
                schema_snapshot[tbl_upper] = {c: {} for c in cols}
            elif tbl_upper in {t.upper() for t in selected_tables}:
                log.warning("[validate_sql] Table %r not in FULL_SCHEMA_CACHE — skipping its column check.", tbl_upper)
            else:
                err = f"Table '{tbl_upper}' not found in schema (not a known table)."
                errors.append(err)
                log.warning("[validate_sql] %s", err)

        if not schema_snapshot:
            log.warning("[validate_sql] No schema data found — skipping validation.")
            return errors

        select_nodes = list(expr.find_all(sqlglot.exp.Select))
        if not select_nodes:
            return errors

        ctx_by_select = {
            id(sel): _SqlResolutionContext(expr, selected_tables, scope_select=sel)
            for sel in select_nodes
        }
        default_ctx = ctx_by_select[id(select_nodes[0])]

        seen: set = set()
        for col in expr.find_all(sqlglot.exp.Column):
            if isinstance(col.this, sqlglot.exp.Star):
                continue
            if not col.table:
                continue
            owning_select = _enclosing_select(col)
            ctx = ctx_by_select.get(id(owning_select), default_ctx) if owning_select else default_ctx

            ref_alias = col.table.lower()
            ref_col = col.name

            pair = (id(ctx), ref_alias, ref_col.lower())
            if pair in seen:
                continue
            seen.add(pair)

            table_name = ctx.alias_to_table.get(ref_alias)
            if table_name is None:
                if ref_alias in ctx.cte_names or ref_alias in ctx.subquery_aliases:
                    continue
                err = (
                    f"Column '{ref_col}' is qualified with '{ref_alias}', which is not a "
                    f"table/alias visible in this query's FROM/JOIN AT THIS SCOPE "
                    f"(and not a CTE or derived-table alias) — "
                    f"this will fail at runtime as \"multipart identifier could not be bound\"."
                )
                errors.append(err)
                log.warning("[validate_sql] %s", err)
                continue

            if table_name not in schema_snapshot:
                continue

            if not ctx.column_exists(table_name, ref_col):
                err = f"Column '{ref_col}' does not exist in table '{table_name}'."
                errors.append(err)
                log.warning("[validate_sql] %s", err)

    except Exception as exc:
        log.warning("[validate_sql] sqlglot parse failed: %s", exc)

    if errors:
        log.warning("[validate_sql] %d validation error(s) for tables %s: %s", len(errors), selected_tables, errors)
    else:
        log.info("[validate_sql] Validation passed for tables %s", selected_tables)
    return errors


def enforce_mandatory_select_columns(
    sql: str,
    live_cols: Dict[str, List[str]],
    llm_tables: List[str],
    join_clause: Optional[str] = None,
) -> str:
    """
    Ensure that every mandatory column (PK + join-FK) that was sent to the LLM
    via the prompt schema actually appears in the generated SQL's SELECT list.

    The LLM sometimes omits PK/FK join columns (e.g. Work_Order_ID) because they
    are not mentioned in the natural-language question. This causes downstream
    issues:
      • GROUP BY / ORDER BY referencing those columns fail.
      • FK translation / LOV resolution cannot operate.
      • Shared-column qualification warnings are triggered even though the fix
        is simply to include the column.

    Algorithm:
      1. Collect mandatory cols per table: actual PKs (from _get_table_pk_cols)
         + join-clause FK cols (from _get_join_fk_cols) that also appear in
         live_cols (i.e. were sent to the LLM in the prompt schema).
      2. Parse the generated SQL; find which table.col pairs are already in SELECT.
      3. For each missing mandatory col, inject it into the SELECT list as
         TableName.ColumnName (fully qualified, no alias).
      4. Return modified SQL (or original if no change / parse error).
    """
    if not sql or not llm_tables or not live_cols:
        return sql

    join_fk_cols: set = _get_join_fk_cols(join_clause)

    required: List[tuple] = []
    for tbl in llm_tables:
        pk_cols = _get_table_pk_cols(tbl)
        prompt_cols_lower = {c.lower(): c for c in live_cols.get(tbl, [])}
        for col_lower, col_orig in prompt_cols_lower.items():
            if col_lower in pk_cols or col_lower in join_fk_cols:
                required.append((tbl, col_orig))

    if not required:
        return sql

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql
        ctx = _SqlResolutionContext(expr, llm_tables, live_cols=live_cols, scope_select=select)

        present: set = set()
        for col_node in select.find_all(sqlglot.exp.Column):
            tbl_ref = (col_node.table or "").lower()
            col_ref = (col_node.name or "").lower()
         
            real_ref = ctx.alias_to_table.get(tbl_ref, tbl_ref.upper()).lower()
            present.add((real_ref, col_ref))
            present.add(("", col_ref))

        injected: List[str] = []
        skipped_not_in_scope: List[str] = []
        for tbl, col_orig in required:
            col_lower = col_orig.lower()
            tbl_lower = tbl.lower()
            if (tbl_lower, col_lower) in present or ("", col_lower) in present:
                continue
            if tbl.upper() not in ctx.real_tables:
                skipped_not_in_scope.append(f"{tbl}.{col_orig}")
                continue
   
            qualifier = ctx.qualifier_for(tbl.upper())
            new_col = sqlglot.exp.Column(
                this=sqlglot.exp.Identifier(this=col_orig),
                table=sqlglot.exp.Identifier(this=qualifier),
            )
            select.append("expressions", new_col)
            present.add((tbl_lower, col_lower))
            injected.append(f"{qualifier}.{col_orig}")

        if skipped_not_in_scope:
            log.info(
                "[enforce_mandatory_cols] Skipped %d mandatory col(s) whose table isn't in "
                "this query's actual FROM/JOIN (avoided dangling qualifier): %s",
                len(skipped_not_in_scope), skipped_not_in_scope,
            )

        if not injected:
            return sql

        result = expr.sql(dialect="tsql")
        log.info(
            "[enforce_mandatory_cols] Injected %d missing mandatory col(s) into SELECT: %s",
            len(injected), injected,
        )
        _vprint(
            "SCHEMA", "cyan",
            f" Mandatory cols injected into SELECT: {injected}",
        )
        return result

    except Exception as exc:
        log.warning("[enforce_mandatory_cols] Failed to inject mandatory cols (%s) — returning original SQL", exc)
        return sql


class SemanticCache:
    _PREFIX_EXACT: str
    _PREFIX_EMB: str
    _INDEX_KEY: str

    def __init__(self, embedder: SentenceTransformer, qdrant: Optional[QdrantClient] = None):
        pfx = _REDIS_KEY_PREFIX
        self._PREFIX_EXACT = f"{pfx}:exact:"
        self._PREFIX_EMB = f"{pfx}:emb:"
        self._INDEX_KEY = f"{pfx}:index"
        self._embedder = embedder
        self._qdrant = qdrant
        self._redis: Optional[redis.Redis] = None
        self._enabled = False
        self._use_qdrant = False
        self._connect()
        if self._enabled and self._qdrant is not None:
            self._init_qdrant_collection()

    def _connect(self) -> None:
        try:
            kwargs = dict(
                host=_REDIS_HOST,
                port=_REDIS_PORT,
                db=_REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            if _REDIS_PASSWORD:
                kwargs["password"] = _REDIS_PASSWORD
            r = redis.Redis(**kwargs)
            r.ping()
            self._redis = r
            self._enabled = True
            log.info(f"[cache] Connected to Redis at {_REDIS_HOST}:{_REDIS_PORT}")
            _vprint("CACHE", "blue", f" Redis connected ({_REDIS_HOST}:{_REDIS_PORT})",
                    f"TTL={_REDIS_TTL}s sim_threshold={_CACHE_SIM_THRESHOLD}")
        except Exception as exc:
            log.warning(f"[cache] Redis unavailable ({exc}) — cache disabled.")

    def _init_qdrant_collection(self) -> None:
        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if _CACHE_QDRANT_COLLECTION not in existing:
                self._qdrant.create_collection(
                    collection_name=_CACHE_QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=_CACHE_EMBED_DIM, distance=Distance.COSINE),
                )
                log.info(f"[cache] Created Qdrant collection '{_CACHE_QDRANT_COLLECTION}'")
            self._use_qdrant = True
            _vprint("CACHE", "blue", f" Qdrant ANN cache ready — collection '{_CACHE_QDRANT_COLLECTION}'")
        except Exception as exc:
            log.warning(f"[cache] Qdrant collection init failed ({exc}) — falling back to Redis linear scan.")

    @staticmethod
    def _cache_key(question: str, tables: List[str]) -> str:
        if _dr_has_current_intent(question):
            _bucket = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        else:
            _bucket = datetime.now(TZ).strftime("%Y-%m-%d")
        raw = question.strip().lower() + "|" + ",".join(sorted(t.lower() for t in tables)) + "|" + _bucket
        return hashlib.sha256(raw.encode()).hexdigest()

    def _embed(self, text: str) -> List[float]:
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _qdrant_search(self, q_vec: List[float], req_tables: set) -> Optional[dict]:
        try:
            hits = None
            last_exc = None
            if hits is None and hasattr(self._qdrant, "query_points"):
                try:
                    result = self._qdrant.query_points(
                        collection_name=_CACHE_QDRANT_COLLECTION,
                        query=q_vec,
                        limit=5,
                        score_threshold=_CACHE_SIM_THRESHOLD,
                        with_payload=True,
                    )
                    hits = result.points if hasattr(result, "points") else result
                except Exception as exc:
                    last_exc = exc
            if hits is None and hasattr(self._qdrant, "search"):
                try:
                    hits = self._qdrant.search(
                        collection_name=_CACHE_QDRANT_COLLECTION,
                        query_vector=q_vec,
                        limit=5,
                        score_threshold=_CACHE_SIM_THRESHOLD,
                        with_payload=True,
                    )
                except Exception as exc:
                    last_exc = exc
            if hits is None:
                if last_exc:
                    log.warning(f"[cache] Qdrant search failed ({last_exc})")
                return None
            for hit in hits:
                stored_tables = set(hit.payload.get("tables", []))
                if stored_tables and not stored_tables.intersection(req_tables):
                    continue
                sha = hit.payload.get("sha")
                if not sha:
                    continue
                raw = self._redis.get(f"{self._PREFIX_EXACT}{sha}")
                if raw:
                    log.info(f"[cache] QDRANT HIT sim={hit.score:.4f} sha={sha[:12]}…")
                    _vprint("CACHE", "cyan", f" QDRANT HIT — sim={hit.score:.4f}")
                    return json.loads(raw)
        except Exception as exc:
            log.warning(f"[cache] Qdrant search failed ({exc}) — falling back to Redis scan.")
        return None

    def _redis_scan(self, q_vec: List[float], req_tables: set) -> Optional[dict]:
        try:
            all_sha = self._redis.lrange(self._INDEX_KEY, 0, -1)
            best_score, best_payload = 0.0, None
            for stored_sha in all_sha:
                emb_raw = self._redis.get(f"{self._PREFIX_EMB}{stored_sha}")
                if not emb_raw:
                    continue
                stored_vec = json.loads(emb_raw)
                score = self._cosine(q_vec, stored_vec)
                if score <= best_score:
                    continue
                candidate_raw = self._redis.get(f"{self._PREFIX_EXACT}{stored_sha}")
                if not candidate_raw:
                    continue
                try:
                    candidate = json.loads(candidate_raw)
                    stored_tables = set(candidate.get("_tables", []))
                    if stored_tables and not stored_tables.intersection(req_tables):
                        continue
                except Exception:
                    continue
                best_score = score
                best_payload = candidate_raw
            if best_score >= _CACHE_SIM_THRESHOLD and best_payload:
                log.info(f"[cache] REDIS SEMANTIC HIT sim={best_score:.4f}")
                _vprint("CACHE", "cyan", f" REDIS HIT — sim={best_score:.4f}")
                return json.loads(best_payload)
        except Exception as exc:
            log.warning(f"[cache] Redis scan error ({exc})")
        return None

    def get(self, question: str, tables: List[str]) -> Optional[dict]:
        if not self._enabled:
            return None
        try:
            sha = self._cache_key(question, tables)
            raw = self._redis.get(f"{self._PREFIX_EXACT}{sha}")
            if raw:
                payload = json.loads(raw)
                payload.pop("_tables", None)
                log.info(f"[cache] EXACT HIT key={sha[:12]}…")
                _vprint("CACHE", "green", " EXACT HIT", f"key={sha[:12]}…")
                if _METRICS_ENABLED:
                    _METRIC_CACHE_HITS.labels(kind="exact").inc()
                return payload

            q_vec = self._embed(question)
            req_tables = {t.lower() for t in tables}

            if self._use_qdrant:
                result = self._qdrant_search(q_vec, req_tables)
                if result is not None and _METRICS_ENABLED:
                    _METRIC_CACHE_HITS.labels(kind="semantic_qdrant").inc()
            else:
                result = self._redis_scan(q_vec, req_tables)
                if result is not None and _METRICS_ENABLED:
                    _METRIC_CACHE_HITS.labels(kind="semantic_redis").inc()

            if result is not None:
                result.pop("_tables", None)
                return result

            log.info("[cache] MISS — will call LLM")
            _vprint("CACHE", "red", " MISS — will call LLM")
            if _METRICS_ENABLED:
                _METRIC_CACHE_MISSES.inc()
            return None
        except Exception as exc:
            log.warning(f"[cache] get() error (non-fatal): {exc}")
            return None

    def set(self, question: str, tables: List[str], payload: dict) -> None:
        if not self._enabled:
            return
        try:
            sha = self._cache_key(question, tables)
            q_vec = self._embed(question)
            pipe = self._redis.pipeline()
            pipe.setex(f"{self._PREFIX_EXACT}{sha}", _REDIS_TTL, json.dumps(payload, default=str))
            if not self._use_qdrant:
                pipe.setex(f"{self._PREFIX_EMB}{sha}", _REDIS_TTL, json.dumps(q_vec))
                pipe.lpush(self._INDEX_KEY, sha)
                pipe.expire(self._INDEX_KEY, _REDIS_TTL)
            pipe.execute()
            if self._use_qdrant:
                point_id = int(sha[:16], 16)
                self._qdrant.upsert(
                    collection_name=_CACHE_QDRANT_COLLECTION,
                    points=[PointStruct(
                        id=point_id,
                        vector=q_vec,
                        payload={"sha": sha, "tables": sorted(t.lower() for t in tables)},
                    )],
                )
            log.info(f"[cache] STORED key={sha[:12]}… qdrant={self._use_qdrant}")
            _vprint("CACHE", "blue", f" STORED key={sha[:12]}… qdrant={self._use_qdrant}")
        except Exception as exc:
            log.warning(f"[cache] set() error (non-fatal): {exc}")

_LLM_CB_FAILURE_THRESHOLD: int = int(os.getenv("LLM_CB_FAILURE_THRESHOLD", "5"))
_LLM_CB_RECOVERY_TIMEOUT: float = float(os.getenv("LLM_CB_RECOVERY_TIMEOUT_SECS", "60"))
_LLM_FALLBACK_MESSAGE: str = os.getenv(
    "LLM_FALLBACK_MESSAGE",
    "The query engine is temporarily unavailable. Please try again in a moment.",
)

class _CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class LLMCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = _LLM_CB_FAILURE_THRESHOLD,
        recovery_timeout: float = _LLM_CB_RECOVERY_TIMEOUT,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._state = _CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float = 0.0
        self._lock = _threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def _trip(self) -> None:
        self._state = _CircuitState.OPEN
        self._opened_at = time.monotonic()
        log.warning(
            "[circuit_breaker] LLM circuit OPENED after %d consecutive failures",
            self._failures,
            extra={"event": "circuit_open", "failures": self._failures},
        )

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == _CircuitState.CLOSED:
                return True
            if self._state == _CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = _CircuitState.HALF_OPEN
                    log.info("[circuit_breaker] Moving to HALF_OPEN — probing LLM server")
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            prev = self._state
            self._state = _CircuitState.CLOSED
            if prev != _CircuitState.CLOSED:
                log.info("[circuit_breaker] LLM circuit CLOSED — server recovered")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == _CircuitState.HALF_OPEN or self._failures >= self._threshold:
                self._trip()
            else:
                log.warning(
                    "[circuit_breaker] LLM failure %d/%d",
                    self._failures, self._threshold,
                    extra={"event": "circuit_failure", "count": self._failures},
                )

_llm_circuit_breaker = LLMCircuitBreaker()

# ============================================================
# MODIFIED NL2SQLPipeline - uses GPU service
# ============================================================

class NL2SQLPipeline:

    def __init__(
        self,
        cache: "SemanticCache",
        graph_resolver: Optional["GraphJoinResolver"] = None,
    ):
        self._cache = cache
        self._graph_resolver = graph_resolver
        log.info("[pipeline] Initialized with GPU service for SQL generation.")

    async def _run_gpu_with_retry(
        self,
        question: str,
        tables: List[str],
        join_clause: Optional[str],
        intent: str,
        module_name: str,
        subgroup_name: str,
        live_cols: Dict[str, List[str]],
        resolved_filters: Optional[Dict[str, str]] = None,
        lov_matches: Optional[Dict[str, Union[str, List[str]]]] = None,
        reverse_hits: Optional[List[dict]] = None,
        module_prompt_rules: str = "",
        status_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Call the GPU service for SQL generation with circuit breaker.
        Returns (sql, intent_override)
        """
        sql, intent_override, retry_count = await _call_gpu_service(
            question=question,
            tables=tables,
            join_clause=join_clause,
            intent=intent,
            module_name=module_name,
            subgroup_name=subgroup_name,
            live_cols=live_cols,
            resolved_filters=resolved_filters,
            lov_matches=lov_matches,
            reverse_hits=reverse_hits,
            module_prompt_rules=module_prompt_rules,
            status_cb=status_cb,
        )
        
        if sql:
            log.info("[pipeline] GPU returned SQL (retries=%d)", retry_count)
            return sql, intent_override
        
        log.error("[pipeline] GPU service failed to generate SQL after retries")
        return None, None

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_db_executor, fn, *args)

    def _execute_sql_sync(
        self, sql: str, params: Optional[tuple] = None
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        _t0 = time.perf_counter()
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(f"SET QUERY_GOVERNOR_COST_LIMIT {_DB_QUERY_TIMEOUT}")
                except Exception:
                    pass
                cur.execute(sql, params) if params else cur.execute(sql)
                rows = cur.fetchall()
                cols_desc = [c[0] for c in cur.description]
                result = [dict(zip(cols_desc, r)) for r in rows]
                if _METRICS_ENABLED:
                    outcome = "zero_rows" if not result else "success"
                    _METRIC_DB_LATENCY.labels(outcome=outcome).observe(time.perf_counter() - _t0)
                return result, None
        except Exception as e:
            log.error("[db] SQL execution failed: %s", e, exc_info=True)
            if _METRICS_ENABLED:
                _METRIC_DB_LATENCY.labels(outcome="error").observe(time.perf_counter() - _t0)
                _METRIC_DB_ERRORS.inc()
            return None, str(e)

    async def _execute_sql_safe(
        self, sql: str, params: Optional[tuple] = None
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_db_executor, self._execute_sql_sync, sql, params)

    async def _execute_sql_with_retry(
        self, sql: str, question: str, all_tables: List[str], intent: str,
        params: Optional[tuple] = None,
        join_clause: Optional[str] = None,
        live_cols: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        rows, err = await self._execute_sql_safe(sql, params)

        if rows is not None and not err:
            rows = _strip_pk_columns(rows, all_tables)

        if err and ("Invalid column name" in err or "Invalid object name" in err):
            log.warning("[pipeline] Column error on first attempt: %s — retrying", err)
            # Note: SQL retries are now handled by GPU, but we keep this for safety
            return rows, err

        if err and "Ambiguous column name" in err:
            log.warning("[pipeline] Ambiguous column error: %s", err)
            _vprint("ALIAS", "red", f" Ambiguous columns — GPU should have handled this")
            return rows, err

        if err and ("8120" in err or "GROUP BY" in err or "aggregate function" in err.lower()):
            log.warning("[pipeline] GROUP BY error — GPU should have handled this")
            return rows, err

        return rows, err

    async def query(
        self,
        question: str,
        tables: List[str],
        module_name: str = "",
        subgroup_name: str = "",
        precomputed_table_hits: Optional[Dict[str, int]] = None,
        force_simple: bool = False,
        llm_intent: Optional[str] = None,
        status_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> dict:
        log.info(f"[pipeline] question={question!r} | tables={tables} | force_simple={force_simple} | llm_intent={llm_intent!r}")
        _vprint("PIPELINE", "magenta", f" Starting pipeline: {question[:80]!r}", f"candidate_tables={tables}")

        cached = self._cache.get(question, tables)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        if not tables:
            return {"error": "No tables provided.", "rows": [], "sql": "", "intent": "simple", "row_count": 0}

        tables = [t for t in tables if t in META]
        if not tables:
            return {"error": "None of the provided tables exist in schema.", "rows": [], "sql": "", "intent": "simple", "row_count": 0}

        log.info("[pipeline] Selected tables after META filter: %s", tables)

        # ── Join-intent detection ────────────────────────────────────────────
        def _pick_intent(heuristic_intent: str) -> str:
            if llm_intent:
                if llm_intent != heuristic_intent:
                    log.info(
                        "[pipeline][intent] llm_intent=%r overrides heuristic_intent=%r",
                        llm_intent, heuristic_intent,
                    )
                return llm_intent
            return heuristic_intent

        base_table = tables[0]
        join_tables: List[str] = list(tables)
        join_clause: Optional[str] = None
        query_type = "simple"
        final_intent = _pick_intent("simple")

        # ── Single-table fast-path ───────────────────────────────────────────
        if force_simple and len(tables) == 1:
            log.info(
                "[pipeline][fast_path] force_simple=True — bypassing all join logic, base=%r",
                base_table,
            )
            _vprint("PIPELINE", "green",
                    f" Fast-path: single table {base_table!r} — join block skipped")

        elif len(tables) > 1:
            if precomputed_table_hits is not None:
                table_hits = dict(precomputed_table_hits)
                log.info("[pipeline][join] Using precomputed table_hits from endpoint.")
            else:
                table_hits = {}
            q_words = set(_extract_question_words(question))
            for tbl in tables:
                tbl_name_words = set(re.findall(r'[a-z]{3,}', tbl.lower()))
                name_hits = len(q_words & tbl_name_words)
                if name_hits:
                    table_hits[tbl] = table_hits.get(tbl, 0) + name_hits * _TABLE_NAME_HIT_WEIGHT

            log.info("[pipeline][join] table_hits (name weighted): %s", table_hits)

            _prebuilt_join_clause: Optional[str] = _has_valid_join_path(tables)
            if _prebuilt_join_clause is not None:
                join_tables = list(tables)
                query_type = "relational"
                final_intent = _pick_intent("relational_lookup")
                join_clause = _prebuilt_join_clause
                log.info(
                    "[pipeline][join] → RELATIONAL join_tables=%s intent=%r (join_clause from graph)",
                    join_tables, final_intent,
                )
            else:
                if table_hits:
                    base_table = sorted(table_hits.items(), key=lambda x: x[1], reverse=True)[0][0]
                join_tables = [base_table]
                final_intent = _pick_intent("simple")
                log.info("[pipeline][join] → SIMPLE (no valid join path in graph), base=%r", base_table)

        log.info(
            "[pipeline] Final table set: base=%r join_tables=%s query_type=%r intent=%r",
            base_table, join_tables, query_type, final_intent,
        )
        _vprint("PIPELINE", "magenta", f" Final table set: {join_tables}", f"query_type={query_type!r} intent={final_intent!r}")

        from core.value_searcher import search_value_in_db as _search_value_in_db

        if status_cb is not None:
            await status_cb("Finding the relevant information...")

        # ── STAGE 0: Reverse-lookup pre-translation ──────────────────────────
        _rl_relevant_tables: List[str] = list(join_tables) if join_tables else (
            [base_table] if base_table else []
        )
        _reverse_hits: List[dict] = _reverse_lookup_question_values(
            question, relevant_tables=_rl_relevant_tables
        )
        _reverse_id_hints: List[str] = []
        _reverse_extra_hint: Optional[str] = None
        _pre_injected_filters: List[dict] = []

        if _reverse_hits:
            _vprint(
                "REVERSE", "green",
                f" Reverse-lookup: {len(_reverse_hits)} phrase(s) resolved",
                " | ".join(f"'{h['phrase']}' → {h['pk_col']}={h['id']}"
                           for h in _reverse_hits),
            )
            for _rh in _reverse_hits:
                _reverse_id_hints.append(
                    f"The phrase '{_rh['phrase']}' refers to column {_rh['pk_col']} = {_rh['id']}. "
                    f"Use the numeric value {_rh['id']} directly in your SQL WHERE clause — "
                    f"never compare a string literal like '{_rh['label']}' against a numeric PK column."
                    f" Do NOT use a subquery to look up this value."
                )
                _pre_injected_filters.append(_rh)

            _reverse_extra_hint = "\n".join(_reverse_id_hints)

        sql: Optional[str] = None
        value = extract_literal_value(question)
        _all_values: List[str] = []
        is_literal_sql = False
        _literal_params: Optional[tuple] = None
        _literal_bridge_table: Optional[str] = None
        _literal_id_hint: Optional[str] = None

        if _reverse_extra_hint:
            _literal_id_hint = _reverse_extra_hint

        if value:
            try:
                _candidates = re.findall(
                    r'\b([A-Za-z]{1,10}[-.]?[A-Za-z0-9]{2,}(?:[-./][A-Za-z0-9]+)*)\b',
                    question,
                )
                _prefix = re.match(r'^[A-Za-z]+', value)
                if _prefix and _candidates:
                    _pfx = _prefix.group(0).upper()
                    _all_values = [
                        v for v in _candidates
                        if v.upper().startswith(_pfx) and v.upper() != _pfx
                    ]
                if not _all_values:
                    _all_values = [value]
            except Exception:
                _all_values = [value]

            log.info("[pipeline] literal=%r all_values=%s", value, _all_values)

       
            _all_meta_tables: List[str] = list(FULL_SCHEMA_CACHE.keys()) or list(META.keys())
            _master_pfxs: tuple = tuple(
                p for p in _TABLE_NAME_PREFIXES
                if not any(p.upper().endswith(s) for s in ("_T_", "_T"))
            ) or tuple(_TABLE_NAME_PREFIXES)

            def _make_where_and_params(tbl: str, col: str):
                if len(_all_values) > 1:
                    _ph = ", ".join("?" * len(_all_values))
                    return (
                        f"UPPER([{tbl}].[{col}]) IN ({_ph})",
                        tuple(v.upper() for v in _all_values),
                    )
                return (
                    f"UPPER([{tbl}].[{col}]) = ?",
                    (value.upper(),),
                )

      
            _CONFUSABLE_PAIRS: tuple = (
                ("0", "O"), ("1", "I"), ("1", "L"), ("5", "S"), ("8", "B"), ("2", "Z"),
            )

            def _fuzzy_variants(v: str, max_variants: int = 12) -> List[str]:
                v_upper = v.upper()
                variants: List[str] = []
                seen = {v_upper}
                for a, b in _CONFUSABLE_PAIRS:
                    for i, ch in enumerate(v_upper):
                        if ch == a:
                            cand = v_upper[:i] + b + v_upper[i + 1:]
                        elif ch == b:
                            cand = v_upper[:i] + a + v_upper[i + 1:]
                        else:
                            continue
                        if cand not in seen:
                            seen.add(cand)
                            variants.append(cand)
                            if len(variants) >= max_variants:
                                return variants
                return variants

            # ── Step 2: search candidate (semantic) tables ────────────────────
            _vprint("LITERAL", "cyan", f" Step 2 — searching candidate tables: {list(tables)}")

            def _db_search_candidates():
                with get_db_connection() as _conn:
                    return _search_value_in_db(
                        _conn,
                        value,
                        candidate_tables=list(tables),
                        all_known_tables=[],
                        master_prefixes=_master_pfxs,
                    )

            owner_table, owner_col = await self._run_in_executor(_db_search_candidates)

            if owner_table:
                _vprint("LITERAL", "green",
                        f" Step 2 hit: {owner_table}.{owner_col} = {value!r}")
            else:
                # ── Step 3: targeted search via Neo4j neighbours ──────────────
                _vprint("LITERAL", "yellow",
                        " Step 2 miss — expanding to master/lookup tables (Step 3)")

                # ── Neo4j activation guard ────────────────────────────────────
                _neo4j_join_intents = {
                    "join", "relational_lookup", "relational_aggregation",
                    "aggregation", "group_by",
                }
                _neo4j_should_activate: bool = (
                    len(tables) > 1
                    and (
                        query_type == "relational"
                        or final_intent in _neo4j_join_intents
                    )
                )

                _neo4j_neighbours: List[str] = []

                if _neo4j_should_activate:
                    log.info(
                        "[pipeline] Step 3 — Neo4j activated "
                        "(multi-table=%d, query_type=%r, intent=%r)",
                        len(tables), query_type, final_intent,
                    )
                    _vprint("LITERAL", "cyan",
                            f" Step 3 — Neo4j graph neighbour lookup "
                            f"(tables={tables}, intent={final_intent!r})")
                    for _t in tables:
                        _neo4j_neighbours.extend(_neo4j_get_neighbor_tables(_t, hops=3))
                    _neo4j_neighbours = [
                        t for t in set(_neo4j_neighbours) - set(tables)
                        if _is_master_table(t)
                    ]
                else:
                    log.info(
                        "[pipeline] Step 3 — Neo4j skipped "
                        "(single-table or simple/filter/status/time query: "
                        "tables=%d, query_type=%r, intent=%r) — going to global scan",
                        len(tables), query_type, final_intent,
                    )
                    _vprint("LITERAL", "yellow",
                            f"⏭ Step 3 — Neo4j skipped "
                            f"(query_type={query_type!r}, intent={final_intent!r}, "
                            f"tables={len(tables)}) → global scan")

                if _neo4j_neighbours:
                    # ── Priority-sort Neo4j neighbours before searching ────────
                    _master_pfxs_sorted = [
                        p for p in _TABLE_NAME_PREFIXES if not p.upper().endswith(("_T_", "_T"))
                    ]
                    def _step3_priority(tbl: str) -> int:
                        t = tbl.upper()
                        for i, pfx in enumerate(_master_pfxs_sorted):
                            if t.startswith(pfx.upper()) and not any(t.endswith(s) for s in _LOG_SUFFIXES):
                                return i
                        return len(_master_pfxs_sorted)

                    _step3_search_tables = sorted(_neo4j_neighbours, key=_step3_priority)
                    log.info(
                        "[pipeline] Step 3 — Neo4j narrowed search to %d table(s) (priority-sorted): %s",
                        len(_step3_search_tables), _step3_search_tables,
                    )
                    _vprint("LITERAL", "cyan",
                            f" Step 3 — Neo4j targeted: {_step3_search_tables}")
                else:
                    _step3_search_tables = _all_meta_tables
                    log.info(
                        "[pipeline] Step 3 — global scan (%d tables)",
                        len(_step3_search_tables),
                    )
                    _vprint("LITERAL", "yellow",
                            f" Step 3 — global scan ({len(_step3_search_tables)} tables)")

                def _db_search_global():
                    with get_db_connection() as _conn:
                        return _search_value_in_db(
                            _conn,
                            value,
                            candidate_tables=[],
                            all_known_tables=_step3_search_tables,
                            master_prefixes=_master_pfxs,
                        )

                owner_table, owner_col = await self._run_in_executor(_db_search_global)

                # ── Step 3 hit validation: if the owner is a log/audit/type
                if owner_table:
                    _owner_upper = owner_table.upper()
                    _is_secondary = any(_owner_upper.endswith(s) for s in _LOG_SUFFIXES)
                    if _is_secondary:
                        _core_candidate = owner_table
                        for _sfx in _LOG_SUFFIXES:
                            if _owner_upper.endswith(_sfx):
                                _core_candidate = owner_table[: len(owner_table) - len(_sfx)]
                                break
                        if (
                            _core_candidate in META
                            and owner_col in META.get(_core_candidate, {}).get("columns", {})
                        ):
                            log.info(
                                "[pipeline] Step 3 — demoting secondary hit %r in favour of "
                                "core master %r (same column %r)",
                                owner_table, _core_candidate, owner_col,
                            )
                            _vprint("LITERAL", "yellow",
                                    f" Step 3 — secondary table {owner_table!r} demoted → "
                                    f"using core master {_core_candidate!r} instead")
                            owner_table = _core_candidate

                if owner_table:
                    _vprint("LITERAL", "green",
                            f" Step 3 hit: {owner_table}.{owner_col} = {value!r}")
                else:
                    # ── Fuzzy retry: look-alike character confusions ──────────
                    _fuzzy_hit_table: Optional[str] = None
                    _fuzzy_hit_col: Optional[str] = None
                    _fuzzy_hit_value: Optional[str] = None
                    _fuzzy_candidates = _fuzzy_variants(value)
                    if _fuzzy_candidates:
                        _vprint("LITERAL", "yellow",
                                f" Step 3 miss — trying {len(_fuzzy_candidates)} "
                                f"look-alike variant(s) of {value!r}")
                    for _fv in _fuzzy_candidates:
                        def _db_search_fuzzy(_val=_fv):
                            with get_db_connection() as _conn:
                                return _search_value_in_db(
                                    _conn,
                                    _val,
                                    candidate_tables=list(tables),
                                    all_known_tables=_step3_search_tables,
                                    master_prefixes=_master_pfxs,
                                )
                        _ft, _fc = await self._run_in_executor(_db_search_fuzzy)
                        if _ft:
                            _fuzzy_hit_table, _fuzzy_hit_col, _fuzzy_hit_value = _ft, _fc, _fv
                            break

                    if _fuzzy_hit_table:
                        log.info(
                            "[pipeline] Fuzzy retry \u2705 — original %r not found, "
                            "matched look-alike %r in %s.%s instead",
                            value, _fuzzy_hit_value, _fuzzy_hit_table, _fuzzy_hit_col,
                        )
                        _vprint("LITERAL", "green",
                                f" Fuzzy retry hit: {_fuzzy_hit_table}.{_fuzzy_hit_col} = "
                                f"{_fuzzy_hit_value!r} (original {value!r} not found)")
                        owner_table, owner_col = _fuzzy_hit_table, _fuzzy_hit_col
                        value = _fuzzy_hit_value
                        _all_values = [_fuzzy_hit_value]
                    else:
                        _vprint("LITERAL", "red", f" Step 3 miss — {value!r} not found anywhere")
                        log.info("[pipeline] Value %r not found in DB — returning no results early.", value)
                        return {
                            "sql": "",
                            "rows": [],
                            "row_count": 0,
                            "intent": final_intent,
                            "error": f"No results found. The value '{value}' was not found in the system. Please verify and try again.",
                            "cache_hit": False,
                            "_tables": sorted(t.lower() for t in tables),
                        }

            # ── Step 4: build SQL from owner table ────────────────────────────
            if owner_table and owner_col:

                # ── Step 4a: owner IS the semantic (base) table ───────────────
                if owner_table.lower() == base_table.lower():
                    _vprint("LITERAL", "cyan",
                            f" Step 4a — same table: {base_table}.{owner_col}")
                    _where, _literal_params = _make_where_and_params(base_table, owner_col)
                    _col_str = f"[{base_table}].*"
                    sql = (
                        f"SELECT TOP {_DEFAULT_TOP} {_col_str} "
                        f"FROM [{base_table}] "
                        f"WHERE {_where}"
                    )
                    is_literal_sql = True
                    # ── Align join_clause: single-table literal needs no JOIN ────
                    join_clause = None
                    log.info("[pipeline] Step-4a SQL: %s | params=%s",
                             sql[:160], _literal_params)
                    _vprint("LITERAL", "green",
                            f" Step 4a done — direct WHERE on {base_table}.{owner_col}")

                # ── Step 4b/c/d: owner is a DIFFERENT table ───────────────────
                else:
                    if owner_table not in join_tables:
                        join_tables.append(owner_table)
                        log.info(
                            "[pipeline] owner_table %r registered in join_tables "
                            "for downstream column validation.",
                            owner_table,
                        )
                        _vprint("LITERAL", "cyan",
                                f" owner_table {owner_table!r} added to join_tables")

                    # ── Step 4b: build_join_clause (FK → BFS → GraphResolver) ──
                    _vprint("LITERAL", "cyan",
                            f" Step 4b — auto-building joins across: {join_tables}")
                    _join_clause_attempt = build_join_clause(
                        base_table, join_tables, intent=final_intent
                    )
                    _join_found = bool(
                        _join_clause_attempt
                        and "JOIN" in _join_clause_attempt.upper()
                    )

                    if _join_found:
                        _where, _literal_params = _make_where_and_params(owner_table, owner_col)
                        _sel_str = f"[{base_table}].*"
                        sql = (
                            f"SELECT TOP {_DEFAULT_TOP} {_sel_str} "
                            f"FROM {_join_clause_attempt} "
                            f"WHERE {_where};"
                        )
                        is_literal_sql = True
                        join_clause = _join_clause_attempt
                        _literal_bridge_table: Optional[str] = owner_table
                        log.info("[pipeline] Step-4b join_tables preserved (full relational set): %s",
                                 join_tables)
                        log.info("[pipeline] Step-4b JOIN SQL: %s | params=%s",
                                 sql[:200], _literal_params)
                        _vprint(
                            "LITERAL", "green",
                            f" Step 4b done — JOIN path: {base_table} ↔ {owner_table}",
                            f"{_join_clause_attempt[:100]}",
                        )

                    else:
                        # ── Step 4c: BFS nested IN(SELECT …) subquery chain ───
                        _vprint("LITERAL", "yellow",
                                f" Step 4b — no direct join found, trying Step 4c BFS subquery")
                        log.info("[pipeline] Step-4b found no join — falling to Step-4c BFS")

                        _JP = _JD.get("join_paths", {})
                        _bfs_path: Optional[list] = None
                        _bfs_q: deque = deque([(base_table, [])])
                        _bfs_seen: set = {base_table.lower()}

                        while _bfs_q and _bfs_path is None:
                            _cur_tbl, _acc = _bfs_q.popleft()
                            if len(_acc) >= _JOIN_MAX_BFS_DEPTH:
                                continue
                            for _next_tbl, _steps in _JP.get(_cur_tbl, {}).items():
                                _new_acc = _acc + [_steps]
                                if _next_tbl.lower() == owner_table.lower():
                                    _bfs_path = [s for seg in _new_acc for s in seg]
                                    break
                                if _next_tbl.lower() not in _bfs_seen:
                                    _bfs_seen.add(_next_tbl.lower())
                                    _bfs_q.append((_next_tbl, _new_acc))

                        if _bfs_path:
                            _owner_pk: Optional[str] = _bfs_path[-1].get("to_column")
                            _base_fk: str = _bfs_path[0]["from_column"]

                            for _fke in get_fk_map().values():
                                if _fke.get("master_table") == owner_table:
                                    _owner_pk = _fke.get("master_pk") or _owner_pk
                                    break

                            _inner = (
                                f"SELECT [{_owner_pk}] FROM [{owner_table}] WITH (NOLOCK) "
                                f"WHERE UPPER([{owner_col}]) = UPPER('{value}')"
                            )
                            for _step in reversed(_bfs_path[1:]):
                                _si = _bfs_path.index(_step)
                                _sel_col = _bfs_path[_si - 1]["to_column"]
                                _inner = (
                                    f"SELECT [{_sel_col}] "
                                    f"FROM [{_step['from_table']}] WITH (NOLOCK) "
                                    f"WHERE [{_step['from_column']}] IN ({_inner})"
                                )

                            _col_str = f"[{base_table}].*"
                            sql = (
                                f"SELECT TOP {_DEFAULT_TOP} {_col_str} "
                                f"FROM [{base_table}] "
                                f"WHERE [{base_table}].[{_base_fk}] IN ({_inner});"
                            )
                            is_literal_sql = True
                            join_clause = None
                            log.info("[pipeline] Step-4c BFS subquery SQL: %s", sql[:200])
                            _vprint(
                                "LITERAL", "green",
                                f" Step 4c done — BFS subquery: {base_table} → {owner_table} "
                                f"({len(_bfs_path)} hop(s))",
                                f"base_fk={_base_fk} owner_col={owner_col}",
                            )

                        else:
                            # ── Step 4d: no path at all → inject numeric PK hint for LLM ──
                            _vprint("LITERAL", "red",
                                    f" Step 4c — no BFS path found, falling to Step 4d (LLM hint)")
                            log.warning(
                                "[pipeline] Step-4d: no join/BFS path %s → %s "
                                "— fetching numeric PK for LLM hint.",
                                base_table, owner_table,
                            )

                            _pk_col: Optional[str] = None
                            for _fke in get_fk_map().values():
                                if _fke.get("master_table") == owner_table:
                                    _pk_col = _fke.get("master_pk")
                                    break
                            if not _pk_col:
                                for _c in META.get(owner_table, {}).get("columns", {}).keys():
                                    if _c.lower().endswith("_id"):
                                        _pk_col = _c
                                        break

                            _num_id: Optional[int] = None
                            if _pk_col:
                                try:
                                    def _fetch_pk():
                                        with get_db_connection() as _c2:
                                            _cur2 = _c2.cursor()
                                            _cur2.execute(
                                                f"SELECT TOP 1 [{_pk_col}] "
                                                f"FROM [{owner_table}] WITH (NOLOCK) "
                                                f"WHERE UPPER([{owner_col}]) = ?",
                                                (value.upper(),),
                                            )
                                            _r = _cur2.fetchone()
                                            return int(_r[0]) if _r else None
                                    _num_id = await self._run_in_executor(_fetch_pk)
                                except Exception as _pke:
                                    log.warning("[pipeline] PK fetch failed: %s", _pke)

                            if _num_id is not None:
                                _literal_id_hint = (
                                    f"The user refers to '{value}'. "
                                    f"Its numeric ID in [{owner_table}] is {_num_id} "
                                    f"(column [{_pk_col}]). "
                                    f"Use this numeric ID directly — never compare a "
                                    f"varchar literal against a numeric column."
                                )
                                log.info("[pipeline] Step-4d hint: %s=%s", _pk_col, _num_id)
                                _vprint("LITERAL", "yellow",
                                        f" Step 4d — LLM hint injected: {owner_table}.{_pk_col} = {_num_id}")
                            else:
                                log.warning(
                                    "[pipeline] Step-4d: PK fetch also failed — "
                                    "LLM will proceed without a numeric hint."
                                )
                                _vprint("LITERAL", "red",
                                        " Step 4d — PK fetch failed, LLM has no hint")

        live_cols = None

        # ── LOV resolution ───────────────────────────────────────────────────
        _UNIVERSAL_RE = re.compile(r'\b(all|every|any|each|list all|show all|get all)\b', re.IGNORECASE)
        _has_universal = bool(_UNIVERSAL_RE.search(question))

        lov_matches: dict = {}
        if not _has_universal:
            all_relevant_tables = [base_table] + join_tables if join_tables else [base_table]
            _unique_lov_tables = list(set(all_relevant_tables))
            _lov_fuzzy_map: Optional[Dict[str, Optional[str]]] = None
            if (
                _global_embedder is not None
                and EmbeddingStore.instance().lov_embed_matrix is not None
            ):
                _q_candidates = list({w for w in re.findall(r'[a-z]{3,}', question.lower())})
                if _q_candidates:
                    _lov_fuzzy_map = EmbeddingStore.instance().batch_fuzzy_lov_match(
                        _q_candidates, _global_embedder
                    )
                    log.info("[pipeline][LOV] Batch fuzzy map: %d words encoded in one call", len(_q_candidates))

            for tbl in _unique_lov_tables:
                tbl_matches = resolve_lov_matches(question, tbl, _precomputed_fuzzy=_lov_fuzzy_map)
                if tbl_matches:
                    lov_matches.update(tbl_matches)

        if sql and lov_matches and value:
            _lov_clauses: List[str] = []
            for _lov_col, _lov_val in lov_matches.items():
                if isinstance(_lov_val, list):
                    _ids = ", ".join(str(v) for v in _lov_val)
                    _lov_clauses.append(f"{base_table}.{_lov_col} IN ({_ids})")
                else:
                    _lov_clauses.append(f"{base_table}.{_lov_col} = {_lov_val}")
            if _lov_clauses:
                sql = sql.rstrip(";") + " AND " + " AND ".join(_lov_clauses)
                log.info(f"[pipeline] LOV filters injected: {_lov_clauses}")
                _vprint("PIPE", "cyan", " LOV + Literal combined", f"filters={_lov_clauses}")

        if sql and value and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"

        if query_type == "relational" and not join_clause and len(join_tables) > 1:
            join_clause = build_join_clause(base_table, join_tables, final_intent)
            if join_clause:
                log.info("[pipeline][join] build_join_clause (fallback) → join_clause=%r", join_clause[:120])
                _vprint("GRAPH", "cyan", f" Join clause built for {join_tables}", join_clause[:100])
            else:
                log.warning(
                    "[pipeline][join] GraphJoinResolver returned no clause for %s → downgrading to simple",
                    join_tables,
                )
                _vprint("GRAPH", "yellow", " No join path found — downgrading to simple", f"tables={join_tables}")
                join_tables = [base_table]
                query_type = "simple"
                final_intent = "simple"
                join_clause = None
        elif query_type == "simple":
            join_clause = None

        llm_tables = list(join_tables) if join_tables else [base_table]

        _is_join_intent = query_type == "relational" or final_intent in (
            "relational_lookup", "relational_aggregation", "join",
        )
        _neo4j_hops = 2 if _is_join_intent else 1

        neighbor_tables: List[str] = get_neighbor_tables(base_table, hops=_neo4j_hops)
        _tx_pfxs = tuple(p for p in _TABLE_NAME_PREFIXES if any(p.upper().endswith(s) for s in ("_T_", "_T")))
        neighbor_tables = [t for t in neighbor_tables if not any(t.upper().startswith(p.upper()) for p in _tx_pfxs)]

        semantic_tables: List[str] = list(llm_tables)
        candidate_tables: List[str] = list(dict.fromkeys(
            [base_table] + neighbor_tables + semantic_tables
        ))
        log.info(
            "[pipeline][neo4j] base=%r hops=%d neighbor_tables=%s "
            "semantic_tables=%s candidate_tables=%s",
            base_table, _neo4j_hops, neighbor_tables, semantic_tables, candidate_tables,
        )
        _vprint("GRAPH", "cyan",
                f" Neo4j candidate set ({_neo4j_hops}-hop): {len(candidate_tables)} table(s)",
                f"candidates={candidate_tables}")

        _all_pipeline_tables = list(candidate_tables)
        if join_clause:
            try:
                _jc_parse = sqlglot.parse_one(
                    f"SELECT 1 FROM {join_clause}",
                    dialect="tsql",
                    error_level=_sqlglot_errors.ErrorLevel.IGNORE,
                )
                for _j in _jc_parse.find_all(sqlglot.exp.Join):
                    _tn = (_j.this.name or "").upper()
                    if _tn and _tn not in {t.upper() for t in _all_pipeline_tables}:
                        _all_pipeline_tables.append(_tn)
                        log.info("[pipeline] Intermediate join table added for live fetch: %s", _tn)
            except Exception:
                pass
        log.info("[pipeline] Full-schema columns for prompt | tables=%s", llm_tables)
        live_cols: Dict[str, List[str]] = {
            tbl: (
                PROMPT_SCHEMA_CACHE.get(tbl, {}).get("columns", [])
                or FULL_SCHEMA_CACHE.get(tbl, {}).get("columns", [])
            )
            for tbl in llm_tables
        }

        if _literal_bridge_table and _literal_bridge_table in live_cols:
            _bridge_pk = _get_table_pk_cols(_literal_bridge_table)
            _bridge_fk = _get_join_fk_cols(join_clause)
            _all_bridge_cols = (
                PROMPT_SCHEMA_CACHE.get(_literal_bridge_table, {}).get("columns", [])
                or FULL_SCHEMA_CACHE.get(_literal_bridge_table, {}).get("columns", [])
            )
            _bridge_mandatory = [
                c for c in _all_bridge_cols
                if c.lower() in _bridge_pk or c.lower() in _bridge_fk
            ]
            if _bridge_mandatory:
                live_cols[_literal_bridge_table] = _bridge_mandatory
                log.info(
                    "[pipeline] Bridge-table %r restricted to mandatory cols only: %s",
                    _literal_bridge_table, _bridge_mandatory,
                )
                _vprint(
                    "SCHEMA", "cyan",
                    f" Bridge table {_literal_bridge_table!r}: only PK/FK cols sent to LLM",
                    f"cols={_bridge_mandatory}",
                )
        log.info(
            "[pipeline] Prompt schema summary: %s",
            {t: len(live_cols.get(t, [])) for t in llm_tables},
        )

        # ── Shared-column detection (scoped to llm_tables only) ───────────────
        shared_cols = get_shared_columns(llm_tables, live_cols)
        if shared_cols:
            log.info("[pipeline] Shared columns detected: %s", list(shared_cols.keys()))
            _vprint("ALIAS", "yellow",
                    f"\u26a0\ufe0f Shared columns found \u2014 full TableName.ColumnName qualification required",
                    f"cols={list(shared_cols.keys())}")

        # ── Unresolved-literal ID hint ────────────────────────────────────────
        if value and not is_literal_sql and not _literal_id_hint:
            _live_cols_for_hint = live_cols.get(base_table, []) if live_cols else []
            _candidate_id_cols = [
                c for c in _live_cols_for_hint
                if re.search(r'work.?order.?id|^wo.?id$', c, re.IGNORECASE)
                   or (c.lower() not in {s.lower() for s in _AUDIT_COL_SUFFIXES}
                       and c.lower().endswith('_id')
                       and any(
                           part in base_table.lower()
                           for part in re.split(r'[_\s]+', c.lower())
                           if len(part) >= 3 and part != 'id'
                       ))
            ]
            if _candidate_id_cols:
                _id_col = _candidate_id_cols[0]
                _literal_id_hint = (
                    f"The value '{value}' is an identifier. "
                    f"Filter using: {base_table}.{_id_col} = '{value}' "
                    f"(do NOT use any other column name for this filter)."
                )
                log.info(
                    "[pipeline] Unresolved literal — injecting ID column hint: %s.%s",
                    base_table, _id_col,
                )

        # ── _prompt_live_cols = live_cols directly (no extra compression) ─────
        _prompt_live_cols = live_cols
        for _tbl, _pcols in _prompt_live_cols.items():
            if _tbl in llm_tables:
                log.info(
                    "[pipeline] PROMPT SCHEMA VERIFIED | table=%r | cols=%d | exact_cols=%s",
                    _tbl, len(_pcols), _pcols,
                )
                _vprint(
                    "SCHEMA", "green",
                    f"\u2705 Verified prompt schema [{_tbl}]: {len(_pcols)} col(s)",
                    f"exact_cols={_pcols}",
                )

        def _query_aware_lov_fn(table: str) -> str:
            return build_query_aware_lov_context(
                table, question, _prompt_live_cols.get(table, [])
            )

        # ── CALL GPU SERVICE ──────────────────────────────────────────────────────
        if not sql: # Only call GPU if we don't already have a literal SQL
            log.info("[pipeline] Calling GPU service for SQL generation …")
            _vprint("GPU", "yellow", " Calling GPU service - SQL generation with XiYanSQL")
            
            # ── Pre-inject resolved reverse-lookup filters ────────────────────
            _skeleton_where_clause: Optional[str] = None
            if _pre_injected_filters:
                _where_parts: List[str] = []
                for _pif in _pre_injected_filters:
                    _id_val = _pif["id"]
                    _pk_col = _pif["pk_col"]
                    _owning_table: Optional[str] = None
                    for _t in llm_tables:
                        _t_cols_lower = [
                            c.lower()
                            for c in (
                                FULL_SCHEMA_CACHE.get(_t, {}).get("columns", [])
                                or PROMPT_SCHEMA_CACHE.get(_t, {}).get("columns", [])
                            )
                        ]
                        if _pk_col.lower() in _t_cols_lower:
                            _owning_table = _t
                            break
                    if _owning_table:
                        _qualified_col = f"{_owning_table}.{_pk_col}"
                    else:
                        _qualified_col = _pk_col

                    if _id_val.lstrip("-").isdigit():
                        _where_parts.append(f"{_qualified_col} = {_id_val}")
                    else:
                        _where_parts.append(f"{_qualified_col} = '{_id_val}'")

                _skeleton_where_clause = " AND ".join(_where_parts)
                log.info(
                    "[pipeline][reverse_inject] Pre-injecting WHERE skeleton: %s",
                    _skeleton_where_clause,
                )
                _vprint(
                    "REVERSE", "cyan",
                    f" Pre-injecting resolved filter into prompt: {_skeleton_where_clause}",
                )

         
            _rule_lookup_key = subgroup_name or module_name
            _module_rule_text = _get_module_prompt_rules(_rule_lookup_key)
            if _rule_lookup_key and not _module_rule_text:
                log.warning(
                    "[pipeline] No prompt rules found for module_name=%r subgroup_name=%r — "
                    "add an entry keyed %r inside a data/module_prompt_rules/*_prompt_rules.json container",
                    module_name, subgroup_name, (_rule_lookup_key or "").lower(),
                )

            _extra_hint_parts: List[str] = []
            if _module_rule_text:
                _extra_hint_parts.append(_module_rule_text)
            if _literal_id_hint:
                _extra_hint_parts.append(_literal_id_hint)
            _combined_extra_hint = "\n\n".join(_extra_hint_parts) if _extra_hint_parts else None

            # Build resolved_filters dict for GPU
            resolved_filters_dict = {}
            if _pre_injected_filters:
                for _pif in _pre_injected_filters:
                    resolved_filters_dict[_pif["pk_col"]] = _pif["id"]
            
            sql, _gpu_intent_override = await self._run_gpu_with_retry(
                question=question,
                tables=llm_tables,
                join_clause=join_clause,
                intent=final_intent,
                module_name=module_name,
                subgroup_name=subgroup_name,
                live_cols=_prompt_live_cols,
                resolved_filters=resolved_filters_dict if resolved_filters_dict else None,
                lov_matches=lov_matches if lov_matches else None,
                reverse_hits=_reverse_hits if _reverse_hits else None,
                module_prompt_rules=_combined_extra_hint or "",
                status_cb=status_cb,
            )
            
            if _gpu_intent_override:
                log.info("[pipeline] GPU overrides intent %r → %r", final_intent, _gpu_intent_override)
                final_intent = _gpu_intent_override
                if final_intent == "simple":
                    join_tables = [base_table]
                    join_clause = None
                    query_type = "simple"

        if not sql:
            return {"error": _LLM_FALLBACK_MESSAGE, "rows": [], "sql": "", "intent": final_intent, "row_count": 0}

        if not sql.strip().upper().startswith("SELECT") and not sql.strip().upper().startswith("WITH"):
            log.warning("[pipeline] GPU returned non-SQL response — rejecting.")
            return {"error": "I wasn't able to generate a valid query for that question. Please try rephrasing.", "rows": [], "sql": "", "intent": final_intent, "row_count": 0}

        sql = post_process_sql(
            sql, base_table, question, final_intent, join_tables, join_clause,
            skip_lov_fix=is_literal_sql,
            shared_cols=shared_cols,
            live_cols=live_cols,
        )
        if not sql:
            log.error("[pipeline] post_process_sql returned empty sentinel — aborting query.")
            return {"error": "I wasn't able to generate a valid query for that question. Please try rephrasing.", "rows": [], "sql": "", "intent": final_intent, "row_count": 0}

        # ── Reverse-lookup SQL fix ────────────────────────────────────────────
        if _reverse_hits:
            sql = _fix_reverse_lookup_in_sql(sql, _reverse_hits)
            if _pre_injected_filters:
                sql = _fix_reverse_lookup_subqueries(sql, _pre_injected_filters)

        # ── Enum literal fix ─────────────────────────────────────────────────
        sql = _fix_enum_literals_by_question(sql, question)

        log.info(f"[pipeline] Final SQL after post_process: {sql}")

        # ── Mandatory column enforcement ──────────────────────────────────────
        if not is_literal_sql:
            sql = _fix_missing_group_by_dimension(
                sql=sql,
                question=question,
                intent=final_intent,
                llm_tables=llm_tables,
                live_cols=live_cols,
            )
            sql = enforce_mandatory_select_columns(
                sql=sql,
                live_cols=live_cols,
                llm_tables=llm_tables,
                join_clause=join_clause,
            )
            sql = _fix_group_by_coherence(sql)

            if _has_pk_only_group_by(sql):
                log.warning("[pipeline] GROUP BY groups only by table's own PK — collapsing to scalar total")
                sql = _collapse_pk_group_by_to_scalar(sql)

            elif _has_spurious_identifier_in_group_by(sql):
                log.warning("[pipeline] GROUP BY mixes a real dimension with a spurious joined-table identifier — stripping it")
                sql = _strip_identifier_cols_from_group_by(sql)
        log.info(f"[pipeline] Final SQL: {sql}")

        # ── Aggregation structure guard ───────────────────────────────────────
        _AGG_INTENTS = {"aggregation", "group_by", "relational_aggregation"}
        if final_intent in _AGG_INTENTS and not is_literal_sql:
            try:
                _agg_check = sqlglot.parse_one(sql, dialect="tsql",
                                               error_level=_sqlglot_errors.ErrorLevel.IGNORE)
                _agg_types = (
                    sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
                    sqlglot.exp.Min, sqlglot.exp.Max,
                    sqlglot.exp.Anonymous,
                )
                _has_agg_fn = any(_agg_check.find(t) is not None for t in _agg_types)
                _has_group = _agg_check.find(sqlglot.exp.Group) is not None
                if not _has_agg_fn and not _has_group:
                    # Note: GPU should have handled this, but we keep as safety net
                    log.warning(
                        "[pipeline][agg_guard] intent=%r but SQL has no agg fn / GROUP BY — "
                        "GPU should have handled this", final_intent
                    )
            except Exception as _agg_exc:
                log.warning("[pipeline][agg_guard] Aggregation guard check failed (%s) — skipping", _agg_exc)

        # ── Detect LLM-generated ──────
        _has_placeholder = bool(re.search(r'\?', sql))
        if _has_placeholder and not is_literal_sql:
            _q_count = sql.count("?")
            if value and _q_count >= 1:
                _bound_val = value.upper()
                _literal_params = tuple([_bound_val] * _q_count)
                is_literal_sql = True
                log.warning(
                    "[pipeline] GPU wrote bare '?' placeholder(s) — binding value %r as param(s)",
                    _bound_val,
                )
                _vprint("LITERAL", "yellow",
                        f" GPU wrote '?' in SQL — auto-binding value {_bound_val!r}",
                        f"placeholders={_q_count}")
            elif _q_count >= 1:
                log.warning(
                    "[pipeline] '?' placeholder in SQL but no literal value available — "
                    "replacing with NULL to prevent parameter binding error"
                )
                sql = sql.replace("?", "NULL")
                _vprint("LITERAL", "red",
                        " Unbound '?' in SQL with no value — replaced with NULL")

        # ── Pre‑execution schema validation ──────────────────────────────────
        _schema_errors = validate_sql(sql, list(set(llm_tables) | {base_table}))
        if _schema_errors:
            _validation_schema = {
                tbl: {col: {} for col in FULL_SCHEMA_CACHE.get(tbl, {}).get("columns", [])}
                for tbl in _all_pipeline_tables
                if FULL_SCHEMA_CACHE.get(tbl, {}).get("columns")
            }
        if _schema_errors:
            _ast_fixed = _ast_fix_column_errors(sql, _schema_errors)
            if _ast_fixed:
                _ast_residual = validate_sql(_ast_fixed, list(set(llm_tables) | {base_table}))
                if not _ast_residual:
                    sql = _ast_fixed
                    log.info("[pipeline][validate] AST fix resolved all errors — skipping LLM fix")
                    _schema_errors = []
                else:
                    log.info("[pipeline][validate] AST fix left %d residual(s) — GPU should have handled this", len(_ast_residual))
                    _schema_errors = _ast_residual
                    sql = _ast_fixed

        if _schema_errors:
            log.warning(
                "[pipeline][validate] %d validation errors — GPU should have handled these: %s",
                len(_schema_errors), _schema_errors
            )
            _vprint("VALIDATE", "red", f" SQL validation errors ({len(_schema_errors)}) — GPU should have fixed these")
            # Return error instead of attempting LLM fix (GPU should have handled it)
            return {
                "error": "I couldn't generate a valid query for that question. Please try rephrasing.",
                "rows": [], "sql": sql, "intent": final_intent, "row_count": 0,
            }

        _exec_params: Optional[tuple] = _literal_params if is_literal_sql else None
        if _exec_params is not None:
            _marker_count = sql.count("?")
            if _marker_count != len(_exec_params):
                log.warning(
                    "[pipeline] Parameter marker mismatch: SQL has %d '?' but "
                    "_literal_params has %d value(s) — clearing params to avoid "
                    "pyodbc HY000 error. sql_fragment=%r",
                    _marker_count, len(_exec_params), sql[:200],
                )
                _vprint(
                    "LITERAL", "yellow",
                    f" Param mismatch ({_marker_count} markers vs "
                    f"{len(_exec_params)} params) — executing without bound params",
                )
                _exec_params = None

        if status_cb is not None:
            await status_cb("Retrieving your results...")

        all_tables_for_retry = list(set(tables + join_tables))
        result_rows, exec_error = await self._execute_sql_with_retry(
            sql, question, all_tables_for_retry, final_intent,
            params=_exec_params,
            join_clause=join_clause,
            live_cols=live_cols,
        )
        if result_rows is None:
            return {"error": "Something went wrong while running the query. Please try again.", "rows": [], "sql": sql, "intent": final_intent, "row_count": 0}

        zero_row_message: Optional[str] = None
        if len(result_rows) == 0:
            log.info('[pipeline] Zero rows returned — surfacing message to user.')
            zero_row_message = "No results found for your query. The data may not exist for these conditions. Try rephrasing with broader criteria."

        # ── Post-execution translation ──────────
        if result_rows:
            try:
                result_rows = translate_rows(result_rows)
                log.info("[pipeline] Post-execution translation applied ")
            except Exception as e:
                log.warning("[pipeline] Post-execution translation failed (non-fatal): %s", e)

        log.info(f"[pipeline] rows returned: {len(result_rows)}")
        _vprint("RESULT", "green", f" {len(result_rows)} row(s) returned | intent={final_intent}", f"SQL: {sql[:90]}…")

        payload = {
            "sql": sql,
            "rows": result_rows,
            "row_count": len(result_rows),
            "intent": final_intent,
            "error": zero_row_message,
            "cache_hit": False,
            "_tables": sorted(t.lower() for t in tables),
        }

        if len(result_rows) > 0:
            self._cache.set(question, tables, payload)
        return payload


# ══════════════════════════════════════════════════════════════════════════════

def _fix_reverse_lookup_in_sql(sql: str, reverse_hits: List[dict]) -> str:
    """
    Post-LLM AST rewrite: replace every string literal in the SQL that matches
    a reverse-lookup display name with the correct numeric (or typed) ID.

    Example:
        Input: WHERE SOME_TABLE.Item_ID = 'Hydraulic Pump Assembly'
        Output: WHERE SOME_TABLE.Item_ID = 1

    Works on EQ predicates and IN lists. Case-insensitive match against the
    display label. Numeric IDs are emitted as number literals; non-numeric IDs
    as string literals.
    """
    if not reverse_hits or not sql:
        return sql
    label_to_id: Dict[str, str] = {h["label"].lower(): h["id"] for h in reverse_hits}
    for h in reverse_hits:
        label_to_id[h["phrase"].lower()] = h["id"]

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
    except Exception as exc:
        log.warning("[fix_reverse_lookup] sqlglot parse failed (%s) — returning sql unchanged", exc)
        return sql

    changed = False

    # ── Fix EQ predicates: col = 'Display Name' ──────────────────────────────
    for eq in expr.find_all(sqlglot.exp.EQ):
        right = eq.right
        if not isinstance(right, sqlglot.exp.Literal) or not right.is_string:
            continue
        val_lower = right.this.lower()
        if val_lower in label_to_id:
            new_id = label_to_id[val_lower]
            if new_id.lstrip("-").isdigit():
                eq.set("right", sqlglot.exp.Literal.number(new_id))
            else:
                eq.set("right", sqlglot.exp.Literal.string(new_id))
            log.info(
                "[fix_reverse_lookup] Rewrote EQ literal %r → %s",
                right.this, new_id,
            )
            changed = True

    # ── Fix IN lists: col IN ('Display Name', ...) ───────────────────────────
    for in_expr in expr.find_all(sqlglot.exp.In):
        new_exprs = []
        in_changed = False
        for item in in_expr.expressions:
            if isinstance(item, sqlglot.exp.Literal) and item.is_string:
                val_lower = item.this.lower()
                if val_lower in label_to_id:
                    new_id = label_to_id[val_lower]
                    if new_id.lstrip("-").isdigit():
                        new_exprs.append(sqlglot.exp.Literal.number(new_id))
                    else:
                        new_exprs.append(sqlglot.exp.Literal.string(new_id))
                    log.info(
                        "[fix_reverse_lookup] Rewrote IN literal %r → %s",
                        item.this, new_id,
                    )
                    in_changed = True
                    changed = True
                    continue
            new_exprs.append(item)
        if in_changed:
            in_expr.set("expressions", new_exprs)

    if not changed:
        return sql

    try:
        fixed = expr.sql(dialect="tsql")
        _vprint(
            "REVERSE", "green",
            f" Reverse-lookup SQL fix applied — {len(reverse_hits)} label(s) → numeric IDs",
        )
        log.info("[fix_reverse_lookup] Fixed SQL: %s", fixed[:200])
        return fixed
    except Exception as exc:
        log.warning("[fix_reverse_lookup] SQL serialisation failed (%s) — returning original", exc)
        return sql


def _fix_reverse_lookup_subqueries(sql: str, pre_injected_filters: List[dict]) -> str:
    """
    Safety-net AST rewrite: if the LLM generated a correlated subquery to look
    up a reverse-lookup value that was already pre-resolved in STAGE 0, replace
    the entire subquery expression with the pre-resolved numeric/typed ID.

    Targets patterns like:
        col = (SELECT pk FROM some_table WHERE name_col = 'Display Label')
        col = (SELECT col FROM table WHERE ...) -- when col matches a pre-resolved pk_col

    This fires only when _pre_injected_filters is non-empty (i.e. STAGE 0 found
    unambiguous hits) and the LLM ignored the pre-injected WHERE directive.
    """
    if not pre_injected_filters or not sql:
        return sql

    # Build pk_col → id map (lowercased key)
    pk_col_to_id: Dict[str, str] = {
        f["pk_col"].lower(): f["id"] for f in pre_injected_filters
    }
    label_to_id: Dict[str, str] = {
        f["label"].lower(): f["id"] for f in pre_injected_filters
    }
    for f in pre_injected_filters:
        label_to_id[f["phrase"].lower()] = f["id"]

    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
    except Exception as exc:
        log.warning("[fix_rl_subqueries] sqlglot parse failed (%s) — returning sql unchanged", exc)
        return sql

    changed = False

    for eq in expr.find_all(sqlglot.exp.EQ):
        left, right = eq.left, eq.right
        if not isinstance(right, sqlglot.exp.Subquery):
            continue
        if not isinstance(left, sqlglot.exp.Column):
            continue

        col_name_lower = left.name.lower()
        resolved_id: Optional[str] = pk_col_to_id.get(col_name_lower)

        if resolved_id is None:

            try:
                sq_where = right.find(sqlglot.exp.Where)
                if sq_where:
                    for sq_eq in sq_where.find_all(sqlglot.exp.EQ):
                        sq_right = sq_eq.right
                        if isinstance(sq_right, sqlglot.exp.Literal) and sq_right.is_string:
                            sq_val_lower = sq_right.this.lower()
                            if sq_val_lower in label_to_id:
                                resolved_id = label_to_id[sq_val_lower]
                                log.info(
                                    "[fix_rl_subqueries] Matched subquery via label literal %r → id=%s",
                                    sq_right.this, resolved_id,
                                )
                                break
            except Exception:
                pass

        if resolved_id is None:
            continue

     
        if resolved_id.lstrip("-").isdigit():
            new_node = sqlglot.exp.Literal.number(resolved_id)
        else:
            new_node = sqlglot.exp.Literal.string(resolved_id)

        eq.set("right", new_node)
        changed = True
        log.info(
            "[fix_rl_subqueries] Replaced subquery for col '%s' with pre-resolved id=%s",
            col_name_lower, resolved_id,
        )
        _vprint(
            "REVERSE", "green",
            f" Subquery replaced: {col_name_lower} = (subquery) → {col_name_lower} = {resolved_id}",
        )

    if not changed:
        return sql

    try:
        fixed = expr.sql(dialect="tsql")
        log.info("[fix_rl_subqueries] Fixed SQL (subquery replaced): %s", fixed[:200])
        return fixed
    except Exception as exc:
        log.warning("[fix_rl_subqueries] SQL serialisation failed (%s) — returning original", exc)
        return sql


# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_global_embedder: Optional[SentenceTransformer] = None
_shared_qdrant_client: Optional[QdrantClient] = None

_audit_executor = ThreadPoolExecutor(max_workers=2)


def log_request_to_db(
    client_ip: str,
    api_key: Optional[str],
    question: str,
    module_name: str,
    subgroup_name: str,
    tables: List[str],
    intent: str,
    sql: str,
    row_count: int,
    error: Optional[str],
    response_time_ms: int,
    cache_hit: bool,
):
    if not _AUDIT_LOG_ENABLED:
        return
    try:
        tables_json = json.dumps(tables) if tables else None
        relaxed_val = 0
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO dbo.chatbot_query_log (
                    client_ip, question, module_name, subgroup_name,
                    tables_used, intent, generated_sql, row_count, error_message,
                    response_time_ms, cache_hit, relaxed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                client_ip, question, module_name or None, subgroup_name or None,
                tables_json, intent, sql, row_count, error, response_time_ms,
                1 if cache_hit else 0, relaxed_val,
            ))
            conn.commit()
    except Exception as e:
        log.error(f"[audit] Failed to insert log: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _graph_resolver, _global_embedder, _shared_qdrant_client

    log.info("[startup] Loading sentence-transformer embedder …")
    embedder = SentenceTransformer(_EMBED_MODEL)
    _global_embedder = embedder
    log.info(f"[startup] Embedder ready ({_EMBED_MODEL}).")
    _vprint("STARTUP", "blue", f" Embedder ready ({_EMBED_MODEL})")

    _store = EmbeddingStore.instance()
    _store.build_phrase_to_label(LOV_DATA)
    _build_table_embeddings(embedder)
    _build_lov_label_embeddings(embedder)
    _store.build_lov_context_cache(LOV_DATA, META, _SYSTEM_STATUS_COLS)
    _vprint("STARTUP", "blue", " EmbeddingStore preloaded — all caches ready")

    _check_readonly_user()

    # ── Translation cache (FK / Type / Headers) ───────────────────────────────
    _translation_ok = load_translation_cache()
    if _translation_ok:
        _vprint("STARTUP", "green", " Translation cache loaded (FK / Type / Headers)")
        # Build reverse-lookup index from the same cache file
        with _reverse_lookup_lock:
            global _REVERSE_LOOKUP_INDEX
            _REVERSE_LOOKUP_INDEX = _build_reverse_lookup_index()
    else:
        _vprint(
            "STARTUP", "yellow",
            " Translation cache missing — run: python scripts/build_master_cache.py",
            "FK/Type IDs will be shown raw until cache is built.",
        )

    # ── LOV translation map — sourced directly from lov_mapping_dynamic.json ──
    _lov_map_ok = _translator_load_lov_map()
    if _lov_map_ok:
        _vprint("STARTUP", "green", " LOV translation map loaded (lov_mapping_dynamic.json)")
    else:
        _vprint(
            "STARTUP", "yellow",
            " lov_mapping_dynamic.json missing — run: python lov_extractor.py",
            "LOV IDs will be shown raw until it's generated.",
        )
    start_cache_refresh_scheduler()
    _vprint("STARTUP", "blue", " Translation cache refresh scheduler started (15-day TTL)")

    _start_chat_history_cleanup_scheduler()
    _vprint("STARTUP", "blue", " Chat history cleanup scheduler started (7-day retention)")

    try:
        async with aiohttp.ClientSession() as sess:
            if _LLM_BACKEND == "ollama":
                health_url = _LLM_SERVER_URL.split("/api/")[0] + "/api/tags"
                async with sess.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        _vprint("STARTUP", "green", " LLM server is reachable")
                    else:
                        _vprint("STARTUP", "yellow", f" LLM server returned {resp.status}")
            else:
                health_url = _LLM_SERVER_URL.replace("/completion", "/health")
                for attempt in range(2):
                    try:
                        async with sess.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            try:
                                body = await resp.json()
                            except Exception:
                                body = {}
                            body_status = str(body.get("status", "")).lower()
                            if resp.status == 200 and body_status in ("ok", ""):
                                _vprint("STARTUP", "green", " LLM server is reachable")
                                break
                            if resp.status == 503 or "loading" in body_status:
                                if attempt == 0:
                                    _vprint("STARTUP", "yellow", "⏳ LLM server still loading model — retrying …")
                                    await asyncio.sleep(3)
                                    continue
                                _vprint("STARTUP", "yellow", f" LLM server still loading model after retry (status={resp.status})")
                            else:
                                _vprint("STARTUP", "yellow", f" LLM server returned {resp.status} (body status={body_status!r})")
                            break
                    except Exception as e:
                        if attempt == 0:
                            await asyncio.sleep(3)
                            continue
                        _vprint("STARTUP", "yellow", f" LLM server not reachable: {e}")
            
            # ── GPU service health check ─────────────────────────────
            try:
                health_url = _GPU_SERVICE_URL.replace("/generate-sql", "/health")
                async with sess.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        _vprint("STARTUP", "green", f" GPU service reachable at {_GPU_SERVICE_URL}")
                    else:
                        _vprint("STARTUP", "yellow", f" GPU service returned {resp.status}")
            except Exception as e:
                _vprint("STARTUP", "yellow", f" GPU service not reachable: {e}")
                
    except Exception as e:
        _vprint("STARTUP", "yellow", f" LLM server not reachable: {e}")

    qdrant_client: Optional[QdrantClient] = None
    if _QDRANT_URL and _QDRANT_API_KEY:
        try:
            qdrant_client = QdrantClient(url=_QDRANT_URL, api_key=_QDRANT_API_KEY, timeout=10)
            _shared_qdrant_client = qdrant_client
            _vprint("STARTUP", "blue", " Qdrant client connected")
        except Exception as e:
            log.warning(f"[startup] Qdrant connection failed: {e}")

    cache = SemanticCache(embedder, qdrant=qdrant_client)

    if cache._enabled:
        _load_rl_script(cache._redis)
        _vprint("STARTUP", "blue", " Lua rate-limit script loaded into Redis")

    log.info("[startup] Initialising GraphJoinResolver …")
    _graph_resolver = GraphJoinResolver(meta=META, join_graph=_JD, embedder=embedder)
    neo4j_ok = _graph_resolver.connect()
    _vprint("STARTUP", "blue" if neo4j_ok else "yellow", f"{'' if neo4j_ok else ''} GraphJoinResolver ready ({'Neo4j' if neo4j_ok else 'JSON-BFS fallback'})")

    _vprint("STARTUP", "blue", " LLM table selector ready (XiYanSQL-7B)")

    log.info("[startup] Loading NL2SQL pipeline …")
    pipeline = NL2SQLPipeline(
        cache=cache,
        graph_resolver=_graph_resolver,
    )
    log.info("[startup] Pipeline ready (GPU service for SQL generation).")
    _vprint("STARTUP", "green", " Pipeline ready — accepting requests")
    yield

    log.info("[shutdown] Shutting down thread pools …")
    _db_executor.shutdown(wait=True)
    _audit_executor.shutdown(wait=True)
    if _graph_resolver:
        _graph_resolver.close()


pipeline: Optional[NL2SQLPipeline] = None

from core.chat_history import (
    save_message as _save_chat_message,
    get_chat_identity as _get_chat_identity,
    start_chat_history_cleanup_scheduler as _start_chat_history_cleanup_scheduler,
)

class QueryRequest(BaseModel):
    question: str
    module: Optional[str] = ""
    subgroup: Optional[str] = ""
    tables: List[str] = []
    conversation_id: Optional[int] = None
    # Set by the frontend only for the very first question sent in a brand
    # new conversation (Title still NULL). Triggers a one-off LLM call to
    # turn the question into a short ChatGPT-style topic title instead of
    # chat_history.py's plain first-80-chars fallback. Never sent (or
    # ignored if stale) for any later message in the same conversation.
    is_new_chat: Optional[bool] = False

class QueryResponse(BaseModel):
    sql: str
    rows: list
    row_count: int
    intent: str
    error: Optional[str] = None
    cache_hit: bool = False
    truncated: bool = False
    debug: Optional[str] = None

# ── Predefined-SQL (sample QA) request model ─────────────────────────────────
class RunSqlRequest(BaseModel):
    sql: str
    module: Optional[str] = ""
    subgroup: Optional[str] = ""

app = FastAPI(title="NL2SQL MES Chatbot", version="3.0", lifespan=lifespan)

from core.chat_history import router as chat_history_router
app.include_router(chat_history_router)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

_templates = Jinja2Templates(directory="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    allow_credentials=False,
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    errors = exc.errors()
    log.error(f"[422] Validation error. Raw body: {body.decode('utf-8', errors='replace')} | Errors: {errors}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": errors,
            "hint": "Required: question (str). Optional: module (str), subgroup (str), tables (list).",
            "body": body.decode("utf-8", errors="replace"),
        },
    )

class _TokenRequest(BaseModel):
    api_key: str

class _TokenResponse(BaseModel):
    token: str
    expires_in: int

@app.post("/auth/token", response_model=_TokenResponse)
async def auth_token(req: _TokenRequest):
    if not secrets.compare_digest(req.api_key.strip(), _API_SECRET_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key.")
    token = _make_session_token()
    log.info("[auth] Session token issued")
    return _TokenResponse(token=token, expires_in=_SESSION_TOKEN_TTL)

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class _QueryEarlyExit(Exception):
    def __init__(self, error: str, status_code: int = 400, intent: str = "error", headers: Optional[dict] = None):
        self.error = error
        self.status_code = status_code
        self.intent = intent
        self.headers = headers
        super().__init__(error)


async def _run_query_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    req: QueryRequest,
):
    start_time = time.perf_counter()

    yield ("status", "Preparing your request...")

    log.info("[endpoint] ▶ QUESTION RECEIVED: %s", req.question)
    _vprint("REQUEST", "magenta", f"▶ {req.question[:100]!r}", f"module={req.module or '—'} subgroup={req.subgroup or '—'}")

    if not req.question.strip():
        yield ("error", {"error": "question cannot be empty", "status_code": 400, "intent": "error"})
        return
    if len(req.question) > _MAX_QUESTION_LEN:
        yield ("error", {"error": f"question exceeds maximum length of {_MAX_QUESTION_LEN} characters", "status_code": 400, "intent": "error"})
        return

    if _looks_like_raw_sql(req.question):
        client_ip_guard = _get_real_ip(request)
        log.warning(
            "[endpoint][sql_guard] BLOCKED raw-SQL-looking input from %s: %r",
            client_ip_guard, req.question[:200],
        )
        yield ("error", {
            "error": "Whoa, Is this a SQL? Just tell me what you need in plain English!",
            "status_code": 400,
            "intent": "error",
        })
        return

    if req.conversation_id:
        try:
            _chat_identity = _get_chat_identity(request)
            _chat_topic = None
            if req.is_new_chat:
                _chat_topic = await _call_chat_topic_llm(req.question)
            _save_chat_message(
                req.conversation_id,
                _chat_identity["username"],
                _chat_identity["plant"],
                "user",
                req.question,
                title_override=_chat_topic,
            )
        except Exception as _chat_save_err:

            log.warning("[chat_history] failed to save user message: %s", _chat_save_err)

    if pipeline is None:
        yield ("error", {"error": "Model is still loading, please retry shortly.", "status_code": 503, "intent": "error"})
        return

    client_ip_rl = _get_real_ip(request)
    if not _check_rate_limit(client_ip_rl):
        yield ("error", {
            "error": "Rate limit exceeded. Please slow down.",
            "status_code": 429,
            "intent": "error",
            "headers": {"Retry-After": str(_RATE_LIMIT_WINDOW)},
        })
        return

    module_key = (req.module or "").strip()
    subgroup_key = (req.subgroup or "").strip()
    flat_tables = [t.strip() for t in (req.tables or []) if t and t.strip()]


    if not module_key or not (subgroup_key or flat_tables):
        yield ("error", {"error": "Provide a valid module (and either a subgroup or a table list) to proceed.", "status_code": 400, "intent": "error"})
        return

    _path_a_hits: Optional[Dict[str, int]] = None

    yield ("status", "Understanding your request...")

    if subgroup_key:
        log.info(f"[endpoint] MODULE SELECTED — '{module_key}/{subgroup_key}' → LLM table selector (XiYanSQL-7B)")
        _vprint("TABLESEL", "blue",
                " Running table + intent selector …",
                f"module={module_key} subgroup={subgroup_key}")
    else:
        log.info(f"[endpoint] MODULE SELECTED — '{module_key}' (no subgroup) → LLM table selector (XiYanSQL-7B) over flat table list ({len(flat_tables)} candidates)")
        _vprint("TABLESEL", "blue",
                " No subgroup config — running table + intent selector over flat candidate list …",
                f"module={module_key} candidates={len(flat_tables)}")

    tables, _llm_selector_intent, _table_selector_error, _, _ = await resolve_tables_llm(
        question=req.question,
        module_name=module_key,
        subgroup_name=subgroup_key,
        flat_table_names=flat_tables if not subgroup_key else None,
    )
    log.info("[endpoint][table_selector_llm] intent=%r", _llm_selector_intent)

    if not tables:
        yield ("error", {"error": _table_selector_error or "Could not determine tables for this question.", "status_code": 400, "intent": "error"})
        return


    log.info(f"[endpoint] Final table set — {len(tables)} table(s): {tables}")
    _vprint("TABLESEL", "green",
            f" Final table set: {len(tables)} table(s) (LLM selection, unmodified)",
            f"tables={tables}")

    # ── Single-table early-exit sentinel ──────────────────────────────────────
    _force_simple: bool = len(tables) == 1
    if _force_simple:
        log.info(
            "[endpoint][single_table] 1 table after filtering — join logic will be bypassed: %r",
            tables[0],
        )
        _vprint("TABLESEL", "green",
                f" Single-table fast-path: {tables[0]!r}",
                "skipping BFS / GraphJoinResolver")

    try:
        await asyncio.wait_for(_request_semaphore.acquire(), timeout=_SEMAPHORE_TIMEOUT)
    except asyncio.TimeoutError:
        yield ("error", {"error": "Server is currently busy. Please try again later.", "status_code": 503, "intent": "error"})
        return

    _status_queue: "asyncio.Queue[str]" = asyncio.Queue()

    async def _status_cb(msg: str) -> None:
        await _status_queue.put(msg)

    try:
        _query_task = asyncio.create_task(pipeline.query(
            question=req.question,
            tables=tables,
            module_name=module_key,
            subgroup_name=subgroup_key,
            precomputed_table_hits=_path_a_hits,
            force_simple=_force_simple,
            llm_intent=_llm_selector_intent,
            status_cb=_status_cb,
        ))

        while not _query_task.done():
            _get_task = asyncio.create_task(_status_queue.get())
            _done, _pending = await asyncio.wait(
                {_query_task, _get_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if _get_task in _done:
                yield ("status", _get_task.result())
            else:
                _get_task.cancel()
                break

      
        while not _status_queue.empty():
            yield ("status", _status_queue.get_nowait())

        result = _query_task.result()
    finally:
        _request_semaphore.release()

    elapsed_secs = time.perf_counter() - start_time
    response_time_ms = int(elapsed_secs * 1000)

    if _METRICS_ENABLED:
        _METRIC_REQUESTS_TOTAL.labels(
            intent=result.get("intent", "unknown"),
            cache_hit="true" if result.get("cache_hit") else "false",
            error="true" if result.get("error") else "false",
        ).inc()
        _METRIC_REQUEST_LATENCY.labels(intent=result.get("intent", "unknown")).observe(elapsed_secs)

    log.info("[endpoint] query complete", extra={
        "event": "query_processed",
        "intent": result.get("intent"),
        "tables": tables,
        "row_count": result.get("row_count", 0),
        "cache_hit": result.get("cache_hit", False),
        "response_time_ms": response_time_ms,
        "error": result.get("error"),
    })

    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get("X-API-Key")
    background_tasks.add_task(
        lambda: _audit_executor.submit(
            log_request_to_db,
            client_ip=client_ip,
            api_key=api_key,
            question=req.question,
            module_name=module_key,
            subgroup_name=subgroup_key,
            tables=tables,
            intent=result.get("intent", ""),
            sql=result.get("sql", ""),
            row_count=result.get("row_count", 0),
            error=result.get("error"),
            response_time_ms=response_time_ms,
            cache_hit=result.get("cache_hit", False),
        )
    )

    result_rows_raw = result.get("rows", [])
    truncated = len(result_rows_raw) > _MAX_RESPONSE_ROWS
    if truncated:
        log.warning("[endpoint] response truncated from %d to %d rows", len(result_rows_raw), _MAX_RESPONSE_ROWS)
    final_rows = result_rows_raw[:_MAX_RESPONSE_ROWS]
    final_row_count = result.get("row_count", len(result_rows_raw))

    yield ("status", "Finalizing your response...")

    if req.conversation_id:
        try:
            _chat_identity = _get_chat_identity(request)
            if result.get("error"):
                _narration = result.get("error")
            else:
                _rc = final_row_count
                _narration = (
                    f"Found {_rc} record{'s' if _rc != 1 else ''} matching your query."
                    if _rc else "No records were found for that query."
                )
           
            _history_rows = final_rows[:_CHAT_HISTORY_MAX_ROWS]
            _history_truncated = truncated or (len(final_rows) > _CHAT_HISTORY_MAX_ROWS)
            _assistant_payload = {
                "narration": _narration,
                "sql": result.get("sql", "") if _SHOW_SQL else "",
                "rows": _history_rows,
                "row_count": final_row_count,
                "intent": result.get("intent", "simple"),
                "error": result.get("error"),
                "cache_hit": result.get("cache_hit", False),
                "truncated": _history_truncated,
            }
            _save_chat_message(
                req.conversation_id,
                _chat_identity["username"],
                _chat_identity["plant"],
                "assistant",
                json.dumps(_assistant_payload, default=str),
            )
        except Exception as _chat_save_err:
            log.warning("[chat_history] failed to save assistant message: %s", _chat_save_err)

    yield ("result", {
        "sql": result.get("sql", "") if _SHOW_SQL else "",
        "rows": final_rows,
        "row_count": final_row_count,
        "intent": result.get("intent", "simple"),
        "error": result.get("error"),
        "cache_hit": result.get("cache_hit", False),
        "truncated": truncated,
        "debug": result.get("debug") if _DEBUG_RESPONSES else None,
    })


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_verify_api_key)])
async def query_endpoint(request: Request, background_tasks: BackgroundTasks, req: QueryRequest):
    async for _kind, _data in _run_query_pipeline(request, background_tasks, req):
        if _kind == "status":
            continue
        if _kind == "error":
            raise HTTPException(
                status_code=_data.get("status_code", 400),
                detail=_data["error"],
                headers=_data.get("headers"),
            )
        # _kind == "result"
        return QueryResponse(**_data)


@app.post("/query/stream", dependencies=[Depends(_verify_api_key)])
async def query_stream_endpoint(request: Request, background_tasks: BackgroundTasks, req: QueryRequest):
    async def _generate():
        async for _kind, _data in _run_query_pipeline(request, background_tasks, req):
            if _kind == "status":
                yield _sse('status', {'message': _data})
            elif _kind == "error":
                yield _sse('error', {
                    'error': _data['error'],
                    'sql': '', 'rows': [], 'row_count': 0,
                    'intent': _data.get('intent', 'error'),
                    'cache_hit': False, 'truncated': False,
                })
                return
            else: # "result"
                _final_payload = {k: v for k, v in _data.items() if k != "debug"}
                yield _sse('done', _final_payload)
                return

    return _FStreamingResponse(_generate(), media_type="text/event-stream")


_RUNSQL_CACHE_PREFIX: str = f"{_REDIS_KEY_PREFIX}:runsql:"

def _runsql_cache_key(sql: str) -> str:
    """Stable SHA256 key for a normalised SQL string."""
    normalised = re.sub(r"\s+", " ", sql.strip().upper())
    return hashlib.sha256(normalised.encode()).hexdigest()


@app.post("/run-sql", dependencies=[Depends(_verify_api_key)])
async def run_sql_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    req: RunSqlRequest,
):
    """
    Execute a predefined SQL query directly — no NL pipeline involved.
    Result is cached in Redis by SQL hash for instant repeat lookups.
    """
    start_time = time.perf_counter()

    # Rate limit — same Redis sliding-window as /query
    client_ip_rl = _get_real_ip(request)
    if not _check_rate_limit(client_ip_rl):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="sql cannot be empty")

    # Safety: only allow SELECT statements — refuse anything else
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT statements are permitted on this endpoint.",
        )

    log.info("[run-sql] ▶ Predefined SQL request | module=%r subgroup=%r | sql=%.120s",
             req.module, req.subgroup, sql)
    _vprint("RUNSQL", "cyan",
            f" /run-sql — predefined query (module={req.module!r} sg={req.subgroup!r})",
            sql[:100])

    # ── 1. Redis cache check ────────────────────────────────────────────────
    cache_hit = False
    cached_rows: Optional[list] = None

    if pipeline is not None and pipeline._cache._enabled:
        try:
            sha = _runsql_cache_key(sql)
            raw = pipeline._cache._redis.get(f"{_RUNSQL_CACHE_PREFIX}{sha}")
            if raw:
                cached_rows = json.loads(raw)
                cache_hit = True
                log.info("[run-sql] CACHE HIT key=%s…", sha[:12])
                _vprint("RUNSQL", "green", f" Cache HIT — {len(cached_rows)} row(s) served instantly",
                        f"key={sha[:12]}…")
                if _METRICS_ENABLED:
                    _METRIC_CACHE_HITS.labels(kind="runsql_exact").inc()
        except Exception as exc:
            log.warning("[run-sql] Cache read failed (non-fatal): %s", exc)

    # ── 2. DB execution (cache miss) ────────────────────────────────────────
    if cached_rows is None:
        _vprint("RUNSQL", "yellow", " Cache MISS — executing SQL against DB …")
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready.")

        # Acquire concurrency slot (same semaphore as /query)
        try:
            await asyncio.wait_for(_request_semaphore.acquire(), timeout=_SEMAPHORE_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Server is currently busy. Please try again.")

        try:
            rows, exec_error = await pipeline._execute_sql_safe(sql)
        finally:
            _request_semaphore.release()

        if rows is None:
            log.error("[run-sql] DB execution failed: %s", exec_error)
            return JSONResponse(
                status_code=200,
                content={
                    "sql": sql if _SHOW_SQL else "",
                    "rows": [],
                    "row_count": 0,
                    "cache_hit": False,
                    "error": "Something went wrong while running the query. Please try again.",
                },
            )

        # ── Post-execution translation (same as /query — LOV→FK→Type→Headers) ──
        if rows:
            try:
                rows = translate_rows(rows)
            except Exception as exc:
                log.warning("[run-sql] Post-execution translation failed (non-fatal): %s", exc)

        cached_rows = rows

        # ── Store in Redis ───────────────────────────────────────────────────
        if pipeline._cache._enabled and rows is not None:
            try:
                sha = _runsql_cache_key(sql)
                pipeline._cache._redis.setex(
                    f"{_RUNSQL_CACHE_PREFIX}{sha}",
                    _REDIS_TTL,
                    json.dumps(rows, default=str),
                )
                log.info("[run-sql] CACHED key=%s… rows=%d ttl=%ds",
                         sha[:12], len(rows), _REDIS_TTL)
                _vprint("RUNSQL", "blue",
                        f" Cached {len(rows)} row(s) for future instant access",
                        f"key={sha[:12]}… ttl={_REDIS_TTL}s")
                if _METRICS_ENABLED:
                    _METRIC_CACHE_MISSES.inc()
            except Exception as exc:
                log.warning("[run-sql] Cache write failed (non-fatal): %s", exc)

    # ── 3. Truncate & respond ───────────────────────────────────────────────
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    truncated = len(cached_rows) > _MAX_RESPONSE_ROWS
    final_rows = cached_rows[:_MAX_RESPONSE_ROWS]

    log.info("[run-sql] %d row(s) returned cache_hit=%s elapsed=%dms",
             len(final_rows), cache_hit, elapsed_ms)
    _vprint("RUNSQL", "green",
            f" {len(final_rows)} row(s) returned (cache_hit={cache_hit} {elapsed_ms}ms)")

    # ── Audit log (background, non-blocking) ────────────────────────────────
    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get("X-API-Key")
    background_tasks.add_task(
        lambda: _audit_executor.submit(
            log_request_to_db,
            client_ip=client_ip,
            api_key=api_key,
            question=f"[SAMPLE] {req.module}/{req.subgroup}",
            module_name=req.module or "",
            subgroup_name=req.subgroup or "",
            tables=[],
            intent="predefined_sql",
            sql=sql,
            row_count=len(final_rows),
            error=None,
            response_time_ms=elapsed_ms,
            cache_hit=cache_hit,
        )
    )

    return {
        "sql": sql if _SHOW_SQL else "",
        "rows": final_rows,
        "row_count": len(cached_rows),
        "cache_hit": cache_hit,
        "truncated": truncated,
        "error": None,
    }


@app.get("/health")
def health(key: str = Security(_api_key_header)):
    authenticated = bool(key and secrets.compare_digest(key, _API_SECRET_KEY))
    cache_status = "disabled"
    cache_qdrant = False
    if pipeline and pipeline._cache._enabled:
        try:
            pipeline._cache._redis.ping()
            cache_status = "connected"
            cache_qdrant = pipeline._cache._use_qdrant
        except Exception:
            cache_status = "error"

    base = {"status": "ok", "timestamp": datetime.now(TZ).isoformat()}
    if not authenticated:
        return base

    graph_resolver_status = {}
    if _graph_resolver:
        graph_resolver_status = _graph_resolver.status()

    base.update({
        "timezone": USER_TIMEZONE,
        "model": Path(MODEL_PATH).name,
        "cache": cache_status,
        "cache_host": f"{_REDIS_HOST}:{_REDIS_PORT}",
        "cache_qdrant_ann": cache_qdrant,
        "sim_threshold": _CACHE_SIM_THRESHOLD,
        "lov_fuzzy_threshold": _LOV_FUZZY_THRESHOLD,
        "llm_server_url": _LLM_SERVER_URL,
        "gpu_service_url": _GPU_SERVICE_URL,
        "llm_circuit_breaker": {
            "state": _llm_circuit_breaker.state,
            "failures": _llm_circuit_breaker._failures,
            "threshold": _llm_circuit_breaker._threshold,
            "recovery_timeout_secs": _llm_circuit_breaker._recovery,
        },
        "graph_resolver": graph_resolver_status,
        "db_pool": _get_pool_status(),
        "concurrency_limit": _MAX_CONCURRENT_REQUESTS,
        "current_semaphore_value": _request_semaphore._value,
        "metrics_enabled": _METRICS_ENABLED,
        "schema_cache": {
            "full_schema_tables": len(FULL_SCHEMA_CACHE),
            "prompt_schema_tables": len(PROMPT_SCHEMA_CACHE),
        },
        "translation_cache": _translation_cache_status(),
        "reverse_lookup_index": {
            "label_count": len(_REVERSE_LOOKUP_INDEX),
            "enabled": len(_REVERSE_LOOKUP_INDEX) > 0,
            "cache_path": str(_MASTER_CACHE_PATH),
        },
    })
    return base

@app.get("/metrics", include_in_schema=False)
def metrics_endpoint(request: Request):
    _verify_metrics_token(request)
    if not _METRICS_ENABLED:
        return JSONResponse(status_code=501, content={"detail": "prometheus-client is not installed."})
    data = generate_latest(REGISTRY)
    return _FResponse(content=data, media_type=CONTENT_TYPE_LATEST)

@app.delete("/cache/clear", dependencies=[Depends(_verify_api_key)])
async def clear_cache():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    if not pipeline._cache._enabled:
        return {"cleared": 0, "message": "Cache is disabled (Redis not connected)."}
    try:
        r = pipeline._cache._redis
        keys = r.keys(f"{_REDIS_KEY_PREFIX}:*")
        if keys:
            r.delete(*keys)
        if pipeline._cache._use_qdrant and _shared_qdrant_client is not None:
            try:
                _shared_qdrant_client.delete_collection(_CACHE_QDRANT_COLLECTION)
                pipeline._cache._init_qdrant_collection()
                log.info("[cache] Qdrant collection cleared and recreated.")
            except Exception as qe:
                log.warning(f"[cache] Qdrant clear failed (non-fatal): {qe}")
        log.info(f"[cache] Manual clear — removed {len(keys)} key(s).")
        return {"cleared": len(keys), "message": "Cache cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {exc}")

@app.post("/admin/reload-lov", dependencies=[Depends(_verify_api_key)])
def reload_lov():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        reload_lov_data()
        return {"status": "LOV data reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.post("/admin/reload-reverse-lookup", dependencies=[Depends(_verify_api_key)])
def reload_reverse_lookup():
    """Hot-reload the reverse-lookup index from master_translation_cache.json."""
    try:
        reload_reverse_lookup_index()
        with _reverse_lookup_lock:
            label_count = len(_REVERSE_LOOKUP_INDEX)
        return {
            "status": "Reverse-lookup index reloaded successfully.",
            "label_count": label_count,
            "cache_path": str(_MASTER_CACHE_PATH),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reverse-lookup reload failed: {e}")

@app.get("/admin/schema-cache/stats", dependencies=[Depends(_verify_api_key)])
def schema_cache_stats():
    return {
        "full_schema_tables": len(FULL_SCHEMA_CACHE),
        "prompt_schema_tables": len(PROMPT_SCHEMA_CACHE),
        "full_schema_path": str(_FULL_SCHEMA_CACHE_PATH),
        "prompt_schema_path": str(_PROMPT_SCHEMA_CACHE_PATH),
        "note": "Static caches — re-run build_schema_cache.py to refresh.",
    }

@app.post("/admin/schema-cache/invalidate", dependencies=[Depends(_verify_api_key)])
def schema_cache_invalidate(table: Optional[str] = None):
    return {
        "status": "not_applicable",
        "message": (
            "Schema caches are static JSON files loaded at startup. "
            "Re-run build_schema_cache.py and restart the server to refresh."
            + (f" (Requested table: {table})" if table else "")
        ),
    }

@app.get("/")
def serve_ui(request: Request):
    return _templates.TemplateResponse(
        _UI_FILENAME,
        {"request": request, "init_key": _API_SECRET_KEY},
    )


@app.get("/data/module_tables.json")
def get_module_tables():
    """
    Serve data/module_tables.json — the single source of truth for all modules.

    Serves data/module_tables.json — the registry of all available modules.
    Add a new module by adding an entry to that file with its config and narrow JSON paths.
    The backend picks it up automatically with no code changes required.
    """
    fpath = _DATA / "module_tables.json"
    if not fpath.exists():
        log.error(
            "[/data/module_tables.json] data/module_tables.json not found. "
            "Create this file to register modules."
        )
        return JSONResponse([])

    try:
        entries: list = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[/data/module_tables.json] Failed to read module_tables.json: %s", exc)
        return JSONResponse([])


    for entry in entries:
        mod_id = (entry.get("id") or "").strip()
        if not mod_id:
            continue
        if entry.get("no_concepts"):
            continue
        if not entry.get("config_file"):
            candidate = _DATA / f"{mod_id.lower()}_module_config.json"
            if candidate.exists():
                entry["config_file"] = f"/data/{mod_id.lower()}_module_config.json"
                log.info(
                    "[/data/module_tables.json] Auto-derived config_file for module %s: %s",
                    mod_id, entry["config_file"],
                )

    log.info(
        "[/data/module_tables.json] Serving %d module(s): %s",
        len(entries),
        [e.get("id") for e in entries],
    )
    return JSONResponse(entries)


@app.get("/data/{filename:path}")
def get_data_file(filename: str):
    # Guard: only allow .json files, no path traversal
    if not filename.endswith(".json") or ".." in filename or "/" in filename:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid filename.")
    fpath = _DATA / filename
    if not fpath.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(str(fpath), media_type="application/json")