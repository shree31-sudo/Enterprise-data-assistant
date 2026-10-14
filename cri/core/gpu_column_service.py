
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.column_prompt_rules import build_column_selection_prompt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gpu_column_service")

# ============================================================
# CONFIG
# ============================================================
_DATA_DIR = Path(os.getenv("COLUMN_SERVICE_DATA_DIR", "./data")).resolve()

_MFC_NARROW_PATH = Path(
    os.getenv("MFC_NARROW_PATH", str(_DATA_DIR / "mfc_narrow.json"))
).resolve()

_FULL_SCHEMA_CACHE_PATH = Path(
    os.getenv("FULL_SCHEMA_CACHE_PATH", str(_DATA_DIR / "full_schema_cache.json"))
).resolve()

_OLLAMA_URL: str = os.getenv(
    "OLLAMA_URL", "http://10.225.30.26:11434/api/generate"  # GPU instance — NOT :11436 (that's CPU: xiyansql14b + qwen3b)
).strip()

_OLLAMA_COLUMN_MODEL: str = os.getenv("OLLAMA_COLUMN_MODEL", "qwen2.5:3b").strip()

_COLUMN_LLM_TIMEOUT: int = int(os.getenv("COLUMN_LLM_TIMEOUT_SECS", "30"))

# ── Reuse the COLUMN_SELECTION_* vars already defined in .env ───────────────
_COLUMN_SELECTION_ENABLED: bool = os.getenv(
    "COLUMN_SELECTION_LLM_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")

_COLUMN_SELECTION_MAX_TABLES: int = int(os.getenv("COLUMN_SELECTION_MAX_TABLES", "3"))
_COLUMN_LLM_MAX_TOKENS: int = int(os.getenv("COLUMN_SELECTION_MAX_TOKENS", "300"))
_COLUMN_DESC_MAX_WORDS: int = int(os.getenv("COLUMN_SELECTION_DESC_MAX_WORDS", "15"))

# ── Queue Config ─────────────────────────────────────────────────────────────
# Same box, same Ollama instance as gpu_service.py (xiyansql) shares the GPU
# with — qwen2.5:3b here runs one generation at a time too, so requests are
# processed strictly FIFO: one /select-columns call finishes before the next
# is picked up. See gpu_service.py for the identical pattern/rationale.
_COLUMN_QUEUE_WORKERS: int = int(os.getenv("COLUMN_QUEUE_WORKERS", "1"))
_COLUMN_QUEUE_MAXSIZE: int = int(os.getenv("COLUMN_QUEUE_MAXSIZE", "20"))
_COLUMN_QUEUE_WAIT_TIMEOUT: float = float(os.getenv("COLUMN_QUEUE_WAIT_TIMEOUT_SECS", "60"))

# ============================================================
# METADATA LOADING  (mfc.json → per-table column catalogue)
# ============================================================
_TABLE_METADATA: Dict[str, Dict] = {}      # table -> {"columns": {col: desc}}
_TABLE_ALL_COLUMNS: Dict[str, List[str]] = {}  # table -> canonical column list (for validation)


def _build_table_metadata_from_narrow(narrow_json: dict) -> Dict[str, Dict]:
    """
    Mirrors the shape produced by chatbot_backend.py's
    _build_narrow_table_index(): {table_name: raw_table_dict}, where
    raw_table_dict["columns"] is {col_name: description}.
    """
    index: Dict[str, Dict] = {}
    for mod_val in narrow_json.get("modules", {}).values():
        for sg_val in mod_val.get("subgroups", {}).values():
            sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
            for tbl in sg_list:
                tname = tbl.get("table_name", "")
                if tname:
                    index[tname] = tbl
    return index


def _load_metadata() -> None:
    global _TABLE_METADATA, _TABLE_ALL_COLUMNS

    if _MFC_NARROW_PATH.exists():
        try:
            narrow_json = json.loads(_MFC_NARROW_PATH.read_text(encoding="utf-8"))
            _TABLE_METADATA = _build_table_metadata_from_narrow(narrow_json)
            log.info(
                "[startup] Loaded mfc_narrow.json: %d tables with column descriptions",
                len(_TABLE_METADATA),
            )
        except Exception as exc:
            log.error("[startup] Failed to parse mfc_narrow.json: %s", exc)
            _TABLE_METADATA = {}
    else:
        log.warning("[startup] mfc_narrow.json not found at %s — descriptions disabled", _MFC_NARROW_PATH)

    # Canonical column whitelist per table, used to validate/repair the
    # model's JSON output so a hallucinated column can never reach SQL gen.
    if _FULL_SCHEMA_CACHE_PATH.exists():
        try:
            full = json.loads(_FULL_SCHEMA_CACHE_PATH.read_text(encoding="utf-8"))
            _TABLE_ALL_COLUMNS = {
                tbl: list(info.get("columns", [])) for tbl, info in full.items()
            }
            log.info(
                "[startup] Loaded full_schema_cache.json: %d tables for column validation",
                len(_TABLE_ALL_COLUMNS),
            )
        except Exception as exc:
            log.error("[startup] Failed to parse full_schema_cache.json: %s", exc)
            _TABLE_ALL_COLUMNS = {}
    else:
        log.warning(
            "[startup] full_schema_cache.json not found at %s — falling back to "
            "narrow-json column keys for validation", _FULL_SCHEMA_CACHE_PATH,
        )
        _TABLE_ALL_COLUMNS = {
            tname: list((meta.get("columns", {}) or {}).keys())
            for tname, meta in _TABLE_METADATA.items()
        }


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="GPU Column Selection Service", version="1.0.0")


# ── Request Queue ────────────────────────────────────────────────────────────
# Same shape as gpu_service.py's queue: one FIFO queue, N workers (default 1,
# matching the single Ollama/qwen2.5:3b instance on this box). A request's
# column-selection work only starts once the one ahead of it in the queue has
# fully finished (built prompt, called qwen, validated/repaired the result).
class _ColumnJob:
    __slots__ = ("req", "future", "enqueued_at")

    def __init__(self, req: "ColumnSelectRequest", future: "asyncio.Future"):
        self.req = req
        self.future = future
        self.enqueued_at = time.monotonic()


_column_queue: "asyncio.Queue[_ColumnJob]" = asyncio.Queue(maxsize=_COLUMN_QUEUE_MAXSIZE)
_column_worker_tasks: List[asyncio.Task] = []


async def _column_worker(worker_id: int) -> None:
    log.info("[queue] column worker-%d started", worker_id)
    while True:
        job = await _column_queue.get()
        try:
            if job.future.cancelled():
                continue
            waited = time.monotonic() - job.enqueued_at
            log.info("[queue] worker-%d picked up job (waited %.1fs, qsize=%d)",
                      worker_id, waited, _column_queue.qsize())
            try:
                result = await _do_select_columns(job.req)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - never let worker die
                log.exception("[queue] worker-%d job failed: %s", worker_id, exc)
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            _column_queue.task_done()


async def _submit_column_job(req: "ColumnSelectRequest") -> "ColumnSelectResponse":
    loop = asyncio.get_running_loop()
    future: "asyncio.Future" = loop.create_future()
    job = _ColumnJob(req, future)

    try:
        _column_queue.put_nowait(job)
    except asyncio.QueueFull:
        log.warning("[queue] rejected job — queue full (maxsize=%d)", _COLUMN_QUEUE_MAXSIZE)
        raise HTTPException(
            status_code=503,
            detail=f"Column service busy — queue full ({_COLUMN_QUEUE_MAXSIZE} pending). Try again shortly.",
        )

    log.info("[queue] job enqueued (position=%d)", _column_queue.qsize())

    try:
        return await asyncio.wait_for(future, timeout=_COLUMN_QUEUE_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        log.error("[queue] job timed out after %.0fs waiting in queue", _COLUMN_QUEUE_WAIT_TIMEOUT)
        raise HTTPException(
            status_code=503,
            detail=f"Column service busy — timed out after {_COLUMN_QUEUE_WAIT_TIMEOUT:.0f}s waiting for a slot.",
        )


@app.on_event("startup")
async def _on_startup() -> None:
    _load_metadata()
    for i in range(_COLUMN_QUEUE_WORKERS):
        _column_worker_tasks.append(asyncio.create_task(_column_worker(i)))
    log.info(f"✅ Started {_COLUMN_QUEUE_WORKERS} column queue worker(s), maxsize={_COLUMN_QUEUE_MAXSIZE}")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    for task in _column_worker_tasks:
        task.cancel()
    for task in _column_worker_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


class ColumnSelectRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    tables: List[str] = Field(..., min_items=1, max_items=20)
    intent: Optional[str] = None


class ColumnSelectResponse(BaseModel):
    columns: Dict[str, List[str]]
    fallback_tables: List[str]  # tables where the model output was rejected/empty
    model: str


def _canonical_columns_for(table: str) -> List[str]:
    return (
        _TABLE_ALL_COLUMNS.get(table)
        or list((_TABLE_METADATA.get(table, {}).get("columns", {}) or {}).keys())
        or []
    )


def _extract_json_object(raw: str) -> Optional[dict]:
    """
    qwen2.5:3b is generally obedient with `format: json`, but this strips
    markdown fences / stray prose defensively before parsing.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost {...} span.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def _call_qwen(prompt: str) -> Optional[str]:
    payload = {
        "model": _OLLAMA_COLUMN_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": _COLUMN_LLM_MAX_TOKENS,
            "temperature": 0.0,
        },
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OLLAMA_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_COLUMN_LLM_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    log.error("[column_llm] Ollama returned HTTP %d", resp.status)
                    return None
                data = await resp.json()
                return (data.get("response") or "").strip()
    except Exception as exc:
        log.error("[column_llm] Call to Ollama failed: %s", exc)
        return None


def _validate_and_repair(
    tables: List[str], parsed: Optional[dict]
) -> tuple[Dict[str, List[str]], List[str]]:
    """
    Whitelist every column the model returned against the real schema.
    Any table the model dropped, mis-cased into oblivion, or returned an
    empty/garbage list for falls back to its FULL column list — never to
    an empty SELECT.
    """
    result: Dict[str, List[str]] = {}
    fallback_tables: List[str] = []

    parsed = parsed or {}
    # Build a case-insensitive lookup of model keys -> actual requested table name
    lower_map = {k.lower(): k for k in parsed.keys()} if isinstance(parsed, dict) else {}

    for tname in tables:
        canonical = _canonical_columns_for(tname)
        canonical_lower = {c.lower(): c for c in canonical}

        model_key = lower_map.get(tname.lower())
        model_cols = parsed.get(model_key, []) if model_key else []

        if not isinstance(model_cols, list):
            model_cols = []

        kept: List[str] = []
        for c in model_cols:
            if not isinstance(c, str):
                continue
            real = canonical_lower.get(c.strip().lower())
            if real and real not in kept:
                kept.append(real)

        if kept:
            result[tname] = kept
        else:
            log.warning(
                "[column_llm] No valid columns for table %r — falling back to full schema", tname
            )
            result[tname] = canonical
            fallback_tables.append(tname)

    return result, fallback_tables


async def _do_select_columns(req: ColumnSelectRequest) -> ColumnSelectResponse:
    """The actual column-selection work (prompt build + qwen call + validate).
    Only ever called from inside a queue worker — never directly from the
    HTTP handler — so FIFO ordering is guaranteed."""
    if not _TABLE_METADATA and not _TABLE_ALL_COLUMNS:
        raise HTTPException(status_code=503, detail="Column metadata not loaded")

    tables = req.tables
    if len(tables) > _COLUMN_SELECTION_MAX_TABLES:
        log.warning(
            "[column_llm] %d tables requested, exceeds COLUMN_SELECTION_MAX_TABLES=%d — truncating",
            len(tables), _COLUMN_SELECTION_MAX_TABLES,
        )
        tables = tables[:_COLUMN_SELECTION_MAX_TABLES]

    if not _COLUMN_SELECTION_ENABLED:
        log.info("[column_llm] COLUMN_SELECTION_LLM_ENABLED=false — full-schema fallback for all tables")
        columns = {t: _canonical_columns_for(t) for t in tables}
        return ColumnSelectResponse(columns=columns, fallback_tables=list(tables), model="disabled")

    prompt = build_column_selection_prompt(
        question=req.question,
        tables=tables,
        table_metadata=_TABLE_METADATA,
        intent=req.intent,
        desc_max_words=_COLUMN_DESC_MAX_WORDS,
    )

    raw = await _call_qwen(prompt)
    parsed = _extract_json_object(raw) if raw else None

    if parsed is None:
        log.warning("[column_llm] Failed to parse model output — full fallback for all tables")

    columns, fallback_tables = _validate_and_repair(tables, parsed)

    return ColumnSelectResponse(
        columns=columns,
        fallback_tables=fallback_tables,
        model=_OLLAMA_COLUMN_MODEL,
    )


@app.post("/select-columns", response_model=ColumnSelectResponse)
async def select_columns(req: ColumnSelectRequest) -> ColumnSelectResponse:
    # Fast-fail checks that don't need GPU/queue time stay here so a bad
    # request doesn't waste a queue slot.
    if not _TABLE_METADATA and not _TABLE_ALL_COLUMNS:
        raise HTTPException(status_code=503, detail="Column metadata not loaded")
    return await _submit_column_job(req)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "enabled": _COLUMN_SELECTION_ENABLED,
        "model": _OLLAMA_COLUMN_MODEL,
        "ollama_url": _OLLAMA_URL,
        "tables_loaded": len(_TABLE_METADATA),
        "schema_tables_loaded": len(_TABLE_ALL_COLUMNS),
        "queue": {
            "pending": _column_queue.qsize(),
            "maxsize": _COLUMN_QUEUE_MAXSIZE,
            "workers": _COLUMN_QUEUE_WORKERS,
            "wait_timeout_secs": _COLUMN_QUEUE_WAIT_TIMEOUT,
        },
    }