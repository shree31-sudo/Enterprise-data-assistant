
from __future__ import annotations

import logging
import os
import threading
import time
from enum import Enum
from typing import Dict, List, Optional

import aiohttp

log = logging.getLogger("column_selector")

# ══════════════════════════════════════════════════════════════════════════
# Config — matches .env exactly, CPU-side names only
# ══════════════════════════════════════════════════════════════════════════
COLUMN_SERVICE_ENABLED: bool = os.getenv("COLUMN_SERVICE_ENABLED", "true").strip().lower() == "true"
COLUMN_SERVICE_URL: str = os.getenv("COLUMN_SERVICE_URL", "http://172.16.1.32:8100/select-columns").strip()
COLUMN_SERVICE_TIMEOUT_SECS: int = int(os.getenv("COLUMN_SERVICE_TIMEOUT_SECS", "35"))

_FALLBACK_COL_SUFFIXES: tuple = tuple(
    s.strip().lower() for s in os.getenv("FALLBACK_COL_SUFFIXES", "_ID").split(",") if s.strip()
)
_FALLBACK_COL_KEYWORDS: tuple = tuple(
    s.strip().lower() for s in os.getenv("FALLBACK_COL_KEYWORDS", "name,code").split(",") if s.strip()
)

_CB_FAILURE_THRESHOLD: int = int(os.getenv("COLUMN_SERVICE_CB_FAILURE_THRESHOLD", "3"))
_CB_RECOVERY_SECS: float = float(os.getenv("COLUMN_SERVICE_CB_RECOVERY_SECS", "30"))


# ══════════════════════════════════════════════════════════════════════════
# Tiny standalone circuit breaker (same shape as chatbot_backend.LLMCircuitBreaker,
# ══════════════════════════════════════════════════════════════════════════
class _CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _ColSelCircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: float) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._state = _CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == _CircuitState.CLOSED:
                return True
            if self._state == _CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = _CircuitState.HALF_OPEN
                    log.info("[column_selector][circuit_breaker] HALF_OPEN — probing column service")
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            prev = self._state
            self._failures = 0
            self._state = _CircuitState.CLOSED
            if prev != _CircuitState.CLOSED:
                log.info("[column_selector][circuit_breaker] CLOSED — column service recovered")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == _CircuitState.HALF_OPEN or self._failures >= self._threshold:
                self._state = _CircuitState.OPEN
                self._opened_at = time.monotonic()
                log.warning("[column_selector][circuit_breaker] OPENED after %d failure(s)", self._failures)


_circuit_breaker = _ColSelCircuitBreaker(_CB_FAILURE_THRESHOLD, _CB_RECOVERY_SECS)



def _heuristic_select(tables: List[str], known_columns: Dict[str, List[str]],
                       mandatory_columns: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    pipeline_cols: Dict[str, List[str]] = {}
    response_cols: Dict[str, List[str]] = {}
    for t in tables:
        real = known_columns.get(t, [])
        mand = set(mandatory_columns.get(t, []))
        keep = set(mand)
        for c in real:
            cl = c.lower()
            if cl.endswith(_FALLBACK_COL_SUFFIXES) or any(kw in cl for kw in _FALLBACK_COL_KEYWORDS):
                keep.add(c)
        p_cols = [c for c in real if c in keep] or list(real)
        pipeline_cols[t] = p_cols
        response_cols[t] = [c for c in p_cols if c not in mand] or p_cols
    return {"pipeline_columns": pipeline_cols, "response_columns": response_cols}


# ══════════════════════════════════════════════════════════════════════════
# Remote call
# ══════════════════════════════════════════════════════════════════════════
async def _call_column_service(payload: dict) -> Optional[dict]:
    if not _circuit_breaker.allow_request():
        log.warning("[column_selector] Circuit breaker OPEN — skipping column service call")
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                COLUMN_SERVICE_URL, json=payload,
                timeout=aiohttp.ClientTimeout(total=COLUMN_SERVICE_TIMEOUT_SECS),
            ) as resp:
                if resp.status != 200:
                    log.error("[column_selector] Column service returned HTTP %d", resp.status)
                    _circuit_breaker.record_failure()
                    return None
                data = await resp.json()
                _circuit_breaker.record_success()
                return data
    except Exception as exc:
        log.error("[column_selector] Column service call failed: %s", exc)
        _circuit_breaker.record_failure()
        return None


# ══════════════════════════════════════════════════════════════════════════
# Public entry point — called from chatbot_backend.py's _select_columns()
# ══════════════════════════════════════════════════════════════════════════
async def select_columns_remote(
    *,
    question: str,
    tables: List[str],
    intent: str,
    module_name: str,
    subgroup_name: str,
    known_columns: Dict[str, List[str]],
    mandatory_columns: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns {"pipeline_columns": {...}, "response_columns": {...}}.
    Every list is already validated against known_columns and has
    mandatory_columns force-included — the caller can use the result as-is.
    """
    mandatory_columns = mandatory_columns or {}
    fallback = {
        "pipeline_columns": {t: list(known_columns.get(t, [])) for t in tables},
        "response_columns": {t: list(known_columns.get(t, [])) for t in tables},
    }

    if not tables:
        return fallback

    if not COLUMN_SERVICE_ENABLED:
        log.info("[column_selector] COLUMN_SERVICE_ENABLED=false — using local heuristic")
        return _heuristic_select(tables, known_columns, mandatory_columns)

    payload = {
        "question": question,
        "tables": tables,
        "module_name": module_name,
        "subgroup_name": subgroup_name,
        "intent": intent,
    }
    data = await _call_column_service(payload)

    if not data or "pipeline_columns" not in data or "response_columns" not in data:
        log.warning("[column_selector] No usable response from column service — using local heuristic")
        return _heuristic_select(tables, known_columns, mandatory_columns)

    def _clean(selected: dict, tbl: str) -> List[str]:
        real = known_columns.get(tbl, [])
        real_lower = {c.lower(): c for c in real}
        picked = selected.get(tbl, []) if isinstance(selected, dict) else []
        if not isinstance(picked, list):
            picked = []
        out: List[str] = []
        for c in picked:
            if isinstance(c, str):
                canon = real_lower.get(c.strip().lower())
                if canon and canon not in out:
                    out.append(canon)
        return out

    pipeline_cols: Dict[str, List[str]] = {}
    response_cols: Dict[str, List[str]] = {}

    for t in tables:
        real_order = known_columns.get(t, [])
        p_cols = _clean(data.get("pipeline_columns", {}), t)

        for m in mandatory_columns.get(t, []):
            if m in real_order and m not in p_cols:
                p_cols.append(m)

        if not p_cols:
            log.warning(
                "[column_selector] Service returned no valid pipeline columns for %r — "
                "falling back to full known column set for this table", t,
            )
            p_cols = list(real_order)

        if real_order:
            p_cols = [c for c in real_order if c in p_cols] + [c for c in p_cols if c not in real_order]

        r_cols = [c for c in _clean(data.get("response_columns", {}), t) if c in p_cols]
        if not r_cols:
            r_cols = list(p_cols)

        pipeline_cols[t] = p_cols
        response_cols[t] = r_cols

    log.info(
        "[column_selector] Selection complete | pipeline=%s | response=%s",
        {t: len(v) for t, v in pipeline_cols.items()},
        {t: len(v) for t, v in response_cols.items()},
    )
    return {"pipeline_columns": pipeline_cols, "response_columns": response_cols}
