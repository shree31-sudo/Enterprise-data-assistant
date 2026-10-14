from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.response_prompt_rules import build_response_shaping_prompt

# build_response_shaping_prompt() is called below with a `column_view:
# Dict[str, str]` kwarg (real_column_name -> plain-English description,
# sourced from the module's narrow-json meta) — core/response_prompt_rules.py
# renders this as a "COLUMN MEANINGS" block in the prompt. The try/except
# around that call further down stays in place as a defensive fallback for
# version-skew deploys (GPU service updated before response_prompt_rules.py,
# or vice versa) rather than something that should trip in normal operation.

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gpu_response_service")

# ============================================================
# CONFIG
# ============================================================
_OLLAMA_URL: str = os.getenv(
    "OLLAMA_URL", "http://10.225.30.26:11434/api/generate"  # GPU instance — NOT :11436 (that's CPU: xiyansql14b + qwen3b)
).strip()

_OLLAMA_RESPONSE_MODEL: str = os.getenv("OLLAMA_RESPONSE_MODEL", "qwen2.5:3b").strip()

_RESPONSE_LLM_TIMEOUT: int = int(os.getenv("RESPONSE_LLM_TIMEOUT_SECS", "30"))

_RESPONSE_SHAPING_ENABLED: bool = os.getenv(
    "RESPONSE_SHAPING_LLM_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")

_RESPONSE_MAX_TOKENS: int = int(os.getenv("RESPONSE_SHAPING_MAX_TOKENS", "550"))
_MAX_SAMPLE_ROWS: int = int(os.getenv("RESPONSE_SHAPING_MAX_SAMPLE_ROWS", "8"))
_LOG_RAW_ON_FAIL: bool = os.getenv(
    "RESPONSE_SHAPING_LOG_RAW_ON_FAIL", "true"
).strip().lower() in ("1", "true", "yes")
_RAW_LOG_CHARS: int = 800

# ── Queue Config ─────────────────────────────────────────────────────────────
# Same GPU/Ollama instance as gpu_service.py and gpu_column_service.py — one
# generation at a time, so requests are processed strictly FIFO: one
# /shape-response call finishes (LLM call + validate/repair) before the next
# is picked up. See gpu_service.py for the identical pattern/rationale.
_RESPONSE_QUEUE_WORKERS: int = int(os.getenv("RESPONSE_QUEUE_WORKERS", "1"))
_RESPONSE_QUEUE_MAXSIZE: int = int(os.getenv("RESPONSE_QUEUE_MAXSIZE", "20"))
_RESPONSE_QUEUE_WAIT_TIMEOUT: float = float(os.getenv("RESPONSE_QUEUE_WAIT_TIMEOUT_SECS", "60"))

# ============================================================
# MODULE META (mfc_narrow.json / usr_narrow.json / ...)
# ============================================================
# The GPU box is a separate process/machine from the CPU pipeline, so it
# keeps its own copy of the module-narrow meta files (same files the CPU
# side loads via core/meta_loader.py -> MODULE_DATA). The CPU pipeline sends
# `module_name` on every /shape-response call; we use it to look up column
# descriptions for that module so qwen2.5:3b gets a "what does this column
# mean" view instead of bare column names, and so we can tell real
# system-generated surrogate primary keys apart from FK/business columns
# that may be exactly what's needed to answer the question.
_META_DATA_DIR: Path = Path(os.getenv("GPU_META_DATA_DIR", "data"))
_NARROW_FILE_GLOB: str = os.getenv("GPU_META_NARROW_GLOB", "*_narrow.json")

# Matches the convention used throughout the narrow-json column
# descriptions for a table's own system-generated surrogate key, e.g.
# "Unique system-generated ID for each item" / "...identifier for this
# item-plant relationship". FK columns pointing at *other* tables use
# different phrasing ("Reference to the item master record") even when
# they share the same physical column name (e.g. Item_ID).
_PK_DESC_PATTERN = re.compile(r"^\s*unique system[- ]generated id", re.IGNORECASE)


def _normalize_field_name(name: str) -> str:
    """
    Normalize field names so that:

    Item Name
    Item_Name
    ITEM-NAME
    item name

    all become

    itemname
    """
    if not isinstance(name, str):
        return ""

    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def _load_module_meta(data_dir: Path) -> Dict[str, dict]:
    """
    Load every *_narrow.json file under `data_dir` (mfc_narrow.json,
    usr_narrow.json, ...) and merge their top-level "modules" dicts into a
    single {module_name: module_dict} map, mirroring what
    core/meta_loader.py::load_all_meta() does on the CPU side. Best-effort:
    a missing/malformed file is logged and skipped rather than crashing the
    service — response shaping degrades to "no column view" for that
    module, it doesn't take the service down.
    """
    merged: Dict[str, dict] = {}
    pattern = str(data_dir / _NARROW_FILE_GLOB)
    paths = sorted(glob.glob(pattern))

    if not paths:
        log.warning(
            "[gpu_meta] No narrow-json meta files found matching %s — "
            "column-view / PK detection disabled until data files are present.",
            pattern,
        )
        return merged

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            modules = raw.get("modules", {})
            if not isinstance(modules, dict):
                log.warning("[gpu_meta] %s: 'modules' is not an object — skipping", p)
                continue
            for mod_key, mod_val in modules.items():
                if mod_key in merged:
                    log.warning(
                        "[gpu_meta] Duplicate module_name %r (in %s) — overwriting "
                        "earlier definition", mod_key, p,
                    )
                merged[mod_key] = mod_val
            log.info("[gpu_meta] Loaded %d module(s) from %s", len(modules), p)
        except Exception as exc:
            log.error("[gpu_meta] Failed to load %s: %s", p, exc)

    return merged


_MODULE_META: Dict[str, dict] = _load_module_meta(_META_DATA_DIR)


def _iter_module_tables(module_name: str):
    """Yield every table dict in a module, across all its subgroups."""
    module = _MODULE_META.get(module_name)
    if not module:
        return
    for subgroup in (module.get("subgroups") or {}).values():
        for table in subgroup.get("tables") or []:
            yield table


def _build_column_context(
    module_name: str, present_columns: List[str]
) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """
    Given the module the question was routed to and the real column names
    present in the row sample, return:

    - `column_view`: {real_column_name: description} for every present
      column we could find a description for, meant to be handed to
      qwen2.5:3b so it knows what each column actually represents instead
      of guessing from the name alone.
    - `raw_pk_columns`: the subset of `present_columns` that are true
      system-generated surrogate primary keys per `_PK_DESC_PATTERN` — a
      column lands here if ANY table in this module describes it as THAT
      TABLE's own row identity (and doesn't list it as one of that same
      table's foreign_keys). This is intentionally OR, not AND: by MES
      naming convention a master table's own PK (Item_ID on MFC_M_ITEM) is
      reused as the FK column name on every table that references it
      (MFC_L_ITEM_PLANT.Item_ID, ...) — so requiring every table to agree
      would mean the PK is *never* detected. The moment even one table's
      own description confirms it's that table's row identity, the column
      must never be surfaced (see the NEVER-SHOW rule in
      response_prompt_rules.py), regardless of it also functioning as an
      FK name elsewhere.
    - `matched_columns`: every present column that meta had *any* opinion
      about (found in at least one table of this module), whether or not
      it ended up in `raw_pk_columns`. Callers use this to tell "meta says
      this is a genuine FK/business column, keep it" apart from "meta
      never saw this column at all" — the latter still falls back to the
      naming heuristic instead of being assumed safe.

    If the module is unknown (missing from _MODULE_META, e.g. the CPU
    pipeline didn't send one, or the module's meta file isn't deployed
    here yet), all three return values are empty and callers fall back to
    showing everything / the legacy naming heuristic.
    """
    column_view: Dict[str, str] = {}
    if not module_name or module_name not in _MODULE_META:
        return column_view, set(), set()

    normalized_present = {_normalize_field_name(c): c for c in present_columns}
    if not normalized_present:
        return column_view, set(), set()

    # Track, per present column, whether ANY table in this module identifies
    # it as ITS OWN system-generated row identity (True) — this must be an
    # OR across tables, not an AND. The MES naming convention is that a
    # master table's own PK (e.g. MFC_M_ITEM.Item_ID) is reused verbatim as
    # the FK column name in every table that references it
    # (MFC_L_ITEM_PLANT.Item_ID, MFC_L_ITEM_PROPERTY.Item_ID, ...). An AND
    # across tables means the very first link table that lists Item_ID in
    # its own foreign_keys flips the vote to "not a PK" everywhere — which
    # is exactly backwards: the moment ANY table's own description says
    # "this is that table's system-generated identifier", the column must
    # never be surfaced, per the NEVER-SHOW rule in
    # response_prompt_rules.py. Being also used as an FK column name
    # elsewhere doesn't change what the column IS when it's playing the
    # master-table-identity role in the current result.
    pk_votes: Dict[str, bool] = {}

    for table in _iter_module_tables(module_name):
        table_columns = table.get("columns") or {}
        fk_names = {_normalize_field_name(fk) for fk in (table.get("foreign_keys") or [])}
        for col_name, desc in table_columns.items():
            norm = _normalize_field_name(col_name)
            real_name = normalized_present.get(norm)
            if not real_name:
                continue  # this table's column isn't in the current row sample

            if real_name not in column_view and isinstance(desc, str) and desc.strip():
                column_view[real_name] = desc.strip()

            is_pk_here = bool(_PK_DESC_PATTERN.match(desc or "")) and norm not in fk_names
            pk_votes[real_name] = pk_votes.get(real_name, False) or is_pk_here

    raw_pk_columns = {name for name, is_pk in pk_votes.items() if is_pk}
    matched_columns = set(pk_votes.keys())  # every present column meta had an opinion on
    return column_view, raw_pk_columns, matched_columns


def _is_column_pk_dropped(
    name: str,
    raw_pk_columns: Set[str],
    matched_columns: Set[str],
) -> bool:
    """
    Decide whether to drop a real column as a raw primary key, using
    meta-driven evidence where we have it (module meta saw this exact
    column somewhere) and falling back to the blunt "*_ID that isn't
    *_VSP_ID" heuristic only for columns meta never saw at all. See
    `_is_primary_key_field` for the legacy fallback and its rationale.
    """
    if name in matched_columns:
        return name in raw_pk_columns
    return _is_primary_key_field(name)


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="GPU Response Shaping Service", version="1.0.0")


# ── Request Queue ────────────────────────────────────────────────────────────
# Same shape as gpu_service.py's / gpu_column_service.py's queue: one FIFO
# queue, N workers (default 1, matching the single Ollama/qwen2.5:3b instance
# on this box). A request's response-shaping work only starts once the one
# ahead of it in the queue has fully finished.
class _ResponseJob:
    __slots__ = ("req", "sample", "future", "enqueued_at")

    def __init__(self, req: "ShapeResponseRequest", sample: List[Dict], future: "asyncio.Future"):
        self.req = req
        self.sample = sample
        self.future = future
        self.enqueued_at = time.monotonic()


_response_queue: "asyncio.Queue[_ResponseJob]" = asyncio.Queue(maxsize=_RESPONSE_QUEUE_MAXSIZE)
_response_worker_tasks: List[asyncio.Task] = []


async def _response_worker(worker_id: int) -> None:
    log.info("[queue] response worker-%d started", worker_id)
    while True:
        job = await _response_queue.get()
        try:
            if job.future.cancelled():
                continue
            waited = time.monotonic() - job.enqueued_at
            log.info("[queue] worker-%d picked up job (waited %.1fs, qsize=%d)",
                      worker_id, waited, _response_queue.qsize())
            try:
                result = await _shape_response_inner(job.req, job.sample)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - never let worker die
                log.exception("[queue] worker-%d job failed: %s", worker_id, exc)
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            _response_queue.task_done()


async def _submit_response_job(req: "ShapeResponseRequest", sample: List[Dict]) -> "ShapeResponseResponse":
    loop = asyncio.get_running_loop()
    future: "asyncio.Future" = loop.create_future()
    job = _ResponseJob(req, sample, future)

    try:
        _response_queue.put_nowait(job)
    except asyncio.QueueFull:
        log.warning("[queue] rejected job — queue full (maxsize=%d)", _RESPONSE_QUEUE_MAXSIZE)
        raise HTTPException(
            status_code=503,
            detail=f"Response service busy — queue full ({_RESPONSE_QUEUE_MAXSIZE} pending). Try again shortly.",
        )

    log.info("[queue] job enqueued (position=%d)", _response_queue.qsize())

    try:
        return await asyncio.wait_for(future, timeout=_RESPONSE_QUEUE_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        log.error("[queue] job timed out after %.0fs waiting in queue", _RESPONSE_QUEUE_WAIT_TIMEOUT)
        raise HTTPException(
            status_code=503,
            detail=f"Response service busy — timed out after {_RESPONSE_QUEUE_WAIT_TIMEOUT:.0f}s waiting for a slot.",
        )


@app.on_event("startup")
async def _on_startup() -> None:
    for i in range(_RESPONSE_QUEUE_WORKERS):
        _response_worker_tasks.append(asyncio.create_task(_response_worker(i)))
    log.info(f"✅ Started {_RESPONSE_QUEUE_WORKERS} response queue worker(s), maxsize={_RESPONSE_QUEUE_MAXSIZE}")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    for task in _response_worker_tasks:
        task.cancel()
    for task in _response_worker_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


class ShapeResponseRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    sql: Optional[str] = ""
    rows: List[Dict] = Field(default_factory=list, max_items=50)
    row_count: int = 0
    intent: Optional[str] = "simple"
    truncated: bool = False
    module_name: Optional[str] = ""


class ShapeResponseResponse(BaseModel):
    fields: List[str]
    narrative: Optional[str] = None
    summary: Optional[str] = None
    model: str


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


def _is_name_field(name: str) -> bool:
    """
    Fields ending in "_Name" (Item_Name, Vendor_Name, Employee_Name, etc.) are
    the primary human-readable label for whatever entity the row is about.
    These are force-included in the output regardless of what the LLM decided
    to keep — dropping the one field that actually identifies the row to a
    human is a worse failure than keeping an extra column.
    """
    if not isinstance(name, str):
        return False
    normalized = _normalize_field_name(name)
    return normalized.endswith("name")


def _is_primary_key_field(name: str) -> bool:
    """
    Blunt fallback heuristic for raw database primary keys, used only for
    columns the module meta file never saw (see `_is_column_pk_dropped`,
    which is the entry point every other function calls). Convention in
    this codebase: genuine lookup/business codes are named "*_VSP_ID" (see
    VSP_ID fallback logic in response_translator.py). A field that ends in
    "_ID" but is NOT a "*_VSP_ID" field is treated as a raw surrogate key.

    This naming-only heuristic is intentionally blunt and will over-drop
    legitimate FK columns — it exists purely as a safety net for when we
    have zero meta information about a column, not as the primary
    decision path.
    """
    if not isinstance(name, str):
        return False
    normalized = _normalize_field_name(name)
    if not normalized.endswith("id"):
        return False
    # Genuine lookup/business codes follow the VSP_ID convention — keep those.
    if normalized.endswith("vspid"):
        return False
    return True


async def _call_qwen(prompt: str) -> Optional[str]:
    payload = {
        "model": _OLLAMA_RESPONSE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": _RESPONSE_MAX_TOKENS,
            "temperature": 0.0,
        },
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OLLAMA_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_RESPONSE_LLM_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    log.error("[response_llm] Ollama returned HTTP %d", resp.status)
                    return None
                data = await resp.json()
                return (data.get("response") or "").strip()
    except Exception as exc:
        log.error("[response_llm] Call to Ollama failed: %s", exc)
        return None


def _validate_and_repair(
    rows: List[Dict],
    parsed: Optional[dict],
    raw: Optional[str] = None,
    raw_pk_columns: Optional[Set[str]] = None,
    matched_columns: Optional[Set[str]] = None,
) -> ShapeResponseResponse:
    """
    Whitelist every field the model returned against the keys that actually
    appear in the row sample. Never let a hallucinated field name reach the
    frontend, and never return an empty field list — fall back to "show
    everything" rather than "show nothing".

    `raw` is the unparsed model output, passed through purely for diagnostic
    logging when something goes wrong (bad JSON, no matching fields) — without
    it, "no valid fields" warnings are undebuggable after the fact.

    `raw_pk_columns` / `matched_columns` come from `_build_column_context`
    (module meta, keyed by the module_name the CPU pipeline sent). When
    present, PK dropping is meta-driven and exact — a real FK/business
    column like Vendor_ID is kept if the query needs it, only genuine
    system-generated surrogate keys get dropped. Columns the module meta
    never saw fall back to the naming heuristic. Pass both as None/empty
    to run purely on the naming heuristic (e.g. unknown module).
    """
    raw_pk_columns = raw_pk_columns or set()
    matched_columns = matched_columns or set()

    def _drop_as_pk(col: str) -> bool:
        return _is_column_pk_dropped(col, raw_pk_columns, matched_columns)

    real_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                real_keys.append(k)

    parsed = parsed or {}
    model_fields = parsed.get("fields", [])
    if not isinstance(model_fields, list):
        model_fields = []

    # Build normalized lookup of the real column names.
    normalized_lookup = {
        _normalize_field_name(k): k
        for k in real_keys
    }

    kept: List[str] = []
    seen = set()

    for field in model_fields:

        if not isinstance(field, str):
            continue

        normalized = _normalize_field_name(field)

        real_field = normalized_lookup.get(normalized)

        if real_field and real_field not in seen and not _drop_as_pk(real_field):
            kept.append(real_field)
            seen.add(real_field)

    dropped_pk = [f for f in real_keys if _drop_as_pk(f)]

    # Always surface name/label columns — the model sometimes omits the one
    # field that actually identifies the row to a human, even though it's
    # exactly what response shaping is supposed to preserve.
    forced_names = [
        k for k in real_keys
        if _is_name_field(k) and not _drop_as_pk(k) and k not in seen
    ]
    for k in forced_names:
        kept.append(k)
        seen.add(k)

    if not kept and rows:
        log.warning("[response_llm] No valid fields in model output — showing all fields")
        if _LOG_RAW_ON_FAIL:
            log.warning(
                "[response_llm] raw model output (first %d chars): %s",
                _RAW_LOG_CHARS,
                (raw or "<empty/no response>")[:_RAW_LOG_CHARS],
            )
            log.warning(
                "[response_llm] parsed fields=%r  real_keys=%r",
                model_fields, real_keys,
            )
        # Even in the "show everything" fallback, never leak raw primary keys.
        kept = [k for k in real_keys if not _drop_as_pk(k)] or real_keys

    narrative = parsed.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        narrative = None

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = None

    # Always-on visibility into what the model proposed vs. what actually
    # made it through normalization/whitelisting — independent of the
    # failure-path logging above, so healthy-looking calls (kept non-empty)
    # are still auditable instead of only logging when things go wrong.
    dropped = [f for f in model_fields if isinstance(f, str) and _normalize_field_name(f) not in normalized_lookup]
    log.info(
        "[response_llm] model_fields=%r kept=%r dropped=%r dropped_primary_keys=%r forced_name_fields=%r narrative=%s summary=%s",
        model_fields,
        kept,
        dropped,
        dropped_pk,
        forced_names,
        "yes" if narrative else "no",
        "yes" if summary else "no",
    )

    return ShapeResponseResponse(
        fields=kept,
        narrative=narrative,
        summary=summary,
        model=_OLLAMA_RESPONSE_MODEL,
    )


@app.post("/shape-response", response_model=ShapeResponseResponse)
async def shape_response(req: ShapeResponseRequest) -> ShapeResponseResponse:
    sample = req.rows[:_MAX_SAMPLE_ROWS]

    if not _RESPONSE_SHAPING_ENABLED:
        real_keys = list({k for r in sample for k in r.keys()})
        return ShapeResponseResponse(fields=real_keys, narrative=None, summary=None, model="disabled")

    if not sample:
        return ShapeResponseResponse(
            fields=[], narrative="No matching records were found.", summary=None,
            model=_OLLAMA_RESPONSE_MODEL,
        )

    try:
        # Queued: this only starts once the request ahead of it has fully
        # finished (LLM call + validate/repair), guaranteeing FIFO order
        # and exactly one Ollama call in flight at a time.
        return await _submit_response_job(req, sample)
    except HTTPException:
        raise
    except Exception as exc:
        # Last-resort safety net: this endpoint is called synchronously in
        # the middle of the CPU pipeline's /query and /run-sql responses
        # (see chatbot_backend.py::_shape_response_via_gpu), and an HTTP 500
        # here used to be indistinguishable from any other failure on that
        # end — it just logs a warning and skips the narrative. Keep it that
        # way for genuinely unexpected errors too, rather than surfacing a
        # 500 that looks like the whole service is down.
        log.error("[response_llm] Unhandled error in /shape-response (non-fatal): %s", exc, exc_info=True)
        real_keys = [k for k in {k for r in sample for k in r.keys()} if not _is_primary_key_field(k)]
        return ShapeResponseResponse(
            fields=real_keys or list({k for r in sample for k in r.keys()}),
            narrative=None, summary=None, model=_OLLAMA_RESPONSE_MODEL,
        )


async def _shape_response_inner(req: ShapeResponseRequest, sample: List[Dict]) -> ShapeResponseResponse:
    present_columns = list({k for r in sample for k in r.keys()})
    try:
        column_view, raw_pk_columns, matched_columns = _build_column_context(
            req.module_name or "", present_columns
        )
    except Exception as exc:
        log.error(
            "[response_llm] _build_column_context failed (non-fatal, "
            "falling back to naming heuristic): %s", exc,
        )
        column_view, raw_pk_columns, matched_columns = {}, set(), set()

    if req.module_name and not matched_columns:
        log.info(
            "[response_llm] module_name=%r had no matching columns in meta "
            "(unknown module, meta file not deployed, or all columns came "
            "from outside this module) — falling back to naming heuristic",
            req.module_name,
        )

    try:
        prompt = build_response_shaping_prompt(
            question=req.question,
            sql=req.sql or "",
            rows=sample,
            row_count=req.row_count,
            intent=req.intent,
            truncated=req.truncated,
            max_sample_rows=_MAX_SAMPLE_ROWS,
            column_view=column_view,
        )
    except TypeError as exc:
        # core/response_prompt_rules.py hasn't been updated to accept
        # `column_view` yet (see NOTE at the top of this file). Don't 500 the
        # whole request over it — fall back to the old prompt shape so
        # response shaping keeps working; column-view/PK smarts just won't
        # reach the model until that file is updated.
        log.warning(
            "[response_llm] build_response_shaping_prompt() rejected "
            "column_view kwarg (%s) — falling back to prompt without column "
            "view. Update core/response_prompt_rules.py to accept it.",
            exc,
        )
        prompt = build_response_shaping_prompt(
            question=req.question,
            sql=req.sql or "",
            rows=sample,
            row_count=req.row_count,
            intent=req.intent,
            truncated=req.truncated,
            max_sample_rows=_MAX_SAMPLE_ROWS,
        )

    raw = await _call_qwen(prompt)
    parsed = _extract_json_object(raw) if raw else None

    if parsed is None:
        log.warning("[response_llm] Failed to parse model output — showing all fields, no narrative")
        if _LOG_RAW_ON_FAIL:
            log.warning(
                "[response_llm] raw model output (first %d chars): %s",
                _RAW_LOG_CHARS,
                (raw or "<empty/no response from Ollama>")[:_RAW_LOG_CHARS],
            )

    return _validate_and_repair(
        sample, parsed, raw=raw,
        raw_pk_columns=raw_pk_columns,
        matched_columns=matched_columns,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "enabled": _RESPONSE_SHAPING_ENABLED,
        "model": _OLLAMA_RESPONSE_MODEL,
        "ollama_url": _OLLAMA_URL,
        "meta_data_dir": str(_META_DATA_DIR),
        "meta_modules_loaded": sorted(_MODULE_META.keys()),
        "queue": {
            "pending": _response_queue.qsize(),
            "maxsize": _RESPONSE_QUEUE_MAXSIZE,
            "workers": _RESPONSE_QUEUE_WORKERS,
            "wait_timeout_secs": _RESPONSE_QUEUE_WAIT_TIMEOUT,
        },
    }