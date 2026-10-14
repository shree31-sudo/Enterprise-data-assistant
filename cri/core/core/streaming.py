from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

log = logging.getLogger("NL2SQL")

# ---------------------------------------------------------------------------
# These names are imported from the host module at registration time so that
# streaming.py carries zero duplicated config.
# ---------------------------------------------------------------------------
_INJECTED: Dict[str, Any] = {}


def _get(name: str) -> Any:
    """Retrieve a value that was injected by register_streaming_routes()."""
    if name not in _INJECTED:
        raise RuntimeError(
            f"[streaming] '{name}' was not injected. "
            "Call register_streaming_routes() before handling requests."
        )
    return _INJECTED[name]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Encode one SSE frame: ``data: <json>\\n\\n``"""
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def _sse_error(detail: str) -> AsyncGenerator[str, None]:
    yield _sse({"event": "error", "detail": detail})


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------

async def _stream_query(
    question: str,
    tables: List[str],
    module_key: str,
    subgroup_key: str,
    precomputed_table_hits: Optional[Dict[str, int]],
) -> AsyncGenerator[str, None]:
    """
    Drive the pipeline and emit SSE events as each stage completes.
    This generator is the heart of the streaming endpoint.
    """
    # -- Unpack injected dependencies ----------------------------------------
    pipeline            = _get("pipeline")
    extract_schema_signals   = _get("extract_schema_signals")
    build_lov_context_block  = _get("build_lov_context_block")
    post_process_sql         = _get("post_process_sql")
    run_llm_async            = _get("run_llm_async")
    build_prompt             = _get("build_prompt")
    META                     = _get("META")
    NARROW_TABLE_INDEX       = _get("NARROW_TABLE_INDEX")
    _MAX_RESPONSE_ROWS       = _get("_MAX_RESPONSE_ROWS")
    _DEBUG_RESPONSES         = _get("_DEBUG_RESPONSES")
    _LLM_MAX_TOKENS_SIMPLE   = _get("_LLM_MAX_TOKENS_SIMPLE")
    _LLM_MAX_TOKENS_COMPLEX  = _get("_LLM_MAX_TOKENS_COMPLEX")
    _LLM_COMPLEX_INTENTS     = _get("_LLM_COMPLEX_INTENTS")
    _request_semaphore       = _get("_request_semaphore")
    _SEMAPHORE_TIMEOUT       = _get("_SEMAPHORE_TIMEOUT")

    start_time = time.perf_counter()

    # ── Event 1: start ───────────────────────────────────────────────────────
    yield _sse({"event": "start", "question": question})

    # ── Event 2: tables ──────────────────────────────────────────────────────
    yield _sse({"event": "tables", "tables": tables})

    # ── Acquire concurrency semaphore ─────────────────────────────────────────
    try:
        await asyncio.wait_for(
            _request_semaphore.acquire(), timeout=_SEMAPHORE_TIMEOUT
        )
    except asyncio.TimeoutError:
        yield _sse({"event": "error", "detail": "Server busy — semaphore timeout."})
        return

    try:
        # ── Check cache first (fast path) ─────────────────────────────────────
        cached = pipeline._cache.get(question, tables)
        if cached:
            rows_raw  = cached.get("rows", [])
            truncated = len(rows_raw) > _MAX_RESPONSE_ROWS
            final_rows = rows_raw[:_MAX_RESPONSE_ROWS]
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield _sse({
                "event":          "cache_hit",
                "sql":            cached.get("sql", ""),
                "rows":           final_rows,
                "row_count":      cached.get("row_count", len(rows_raw)),
                "intent":         cached.get("intent", "simple"),
                "truncated":      truncated,
                "cache_hit":      True,
                "response_time_ms": elapsed_ms,
                "debug":          cached.get("debug") if _DEBUG_RESPONSES else None,
            })
            yield _sse({"event": "done", "cache_hit": True, "response_time_ms": elapsed_ms})
            return

        # ── LLM path: build prompt ────────────────────────────────────────────
        # We replicate only the prompt-build + LLM call stages here so that we
        # can emit the "sql" event as soon as the LLM returns, before DB exec.
        # The full pipeline.query() call handles the rest (post-processing, DB).
        # We therefore call pipeline.query() for post-processing + DB, but emit
        # intermediate events around it.

        # Emit a "thinking" event so the UI can show a spinner immediately.
        yield _sse({"event": "thinking"})

        # Delegate to pipeline.query() for the full result — this keeps all
        # the complex post-processing, join resolution, LOV fixups, etc. in
        # one place. We intercept *after* it returns to split the response.
        result = await pipeline.query(
            question=question,
            tables=tables,
            module_name=module_key,
            subgroup_name=subgroup_key,
            precomputed_table_hits=precomputed_table_hits,
        )

        # ── Event 3: sql ─────────────────────────────────────────────────────
        if result.get("sql"):
            yield _sse({"event": "sql", "sql": result["sql"]})

        # ── Event 4: rows ─────────────────────────────────────────────────────
        rows_raw  = result.get("rows", [])
        truncated = len(rows_raw) > _MAX_RESPONSE_ROWS
        final_rows = rows_raw[:_MAX_RESPONSE_ROWS]
        yield _sse({
            "event":     "rows",
            "rows":      final_rows,
            "row_count": result.get("row_count", len(rows_raw)),
            "truncated": truncated,
        })

        # ── Event 5: done ─────────────────────────────────────────────────────
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        yield _sse({
            "event":            "done",
            "intent":           result.get("intent", "simple"),
            "cache_hit":        result.get("cache_hit", False),
            "response_time_ms": elapsed_ms,
            "error":            result.get("error"),
            "debug":            result.get("debug") if _DEBUG_RESPONSES else None,
        })

    except Exception as exc:
        log.error("[streaming] Unhandled error in stream generator: %s", exc, exc_info=True)
        yield _sse({"event": "error", "detail": str(exc)})
    finally:
        _request_semaphore.release()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class StreamQueryRequest(BaseModel):
    question: str
    module:   Optional[str] = ""
    subgroup: Optional[str] = ""
    tables:   List[str] = []


# ---------------------------------------------------------------------------
# Public registration function
# ---------------------------------------------------------------------------

def register_streaming_routes(
    app: FastAPI,
    *,
    # All of these default to None so callers can pass them explicitly.
    # register_streaming_routes() resolves them via the injected dict when None.
    pipeline=None,
    extract_schema_signals=None,
    build_lov_context_block=None,
    post_process_sql=None,
    run_llm_async=None,
    build_prompt=None,
    META=None,
    NARROW_TABLE_INDEX=None,
    _MAX_RESPONSE_ROWS=None,
    _DEBUG_RESPONSES=None,
    _LLM_MAX_TOKENS_SIMPLE=None,
    _LLM_MAX_TOKENS_COMPLEX=None,
    _LLM_COMPLEX_INTENTS=None,
    _request_semaphore=None,
    _SEMAPHORE_TIMEOUT=None,
    # Security helpers
    api_key_header: Optional[APIKeyHeader] = None,
    api_secret_key: str = "",
    # Rate-limit helpers
    get_real_ip=None,
    check_rate_limit=None,
    rate_limit_window: int = 60,
    # Table resolution helper
    resolve_tables_hybrid=None,
    max_question_len: int = 1000,
) -> None:
    """
    Wire ``POST /query/stream`` into *app*.

    Call this once during startup, after the pipeline is initialised:

        import secrets as _secrets
        from core.streaming import register_streaming_routes

        register_streaming_routes(
            app,
            pipeline=pipeline,
            extract_schema_signals=extract_schema_signals,
            build_lov_context_block=build_lov_context_block,
            post_process_sql=post_process_sql,
            run_llm_async=run_llm_async,
            build_prompt=_ext_build_prompt,
            META=META,
            NARROW_TABLE_INDEX=NARROW_TABLE_INDEX,
            _MAX_RESPONSE_ROWS=_MAX_RESPONSE_ROWS,
            _DEBUG_RESPONSES=_DEBUG_RESPONSES,
            _LLM_MAX_TOKENS_SIMPLE=_LLM_MAX_TOKENS_SIMPLE,
            _LLM_MAX_TOKENS_COMPLEX=_LLM_MAX_TOKENS_COMPLEX,
            _LLM_COMPLEX_INTENTS=_LLM_COMPLEX_INTENTS,
            _request_semaphore=_request_semaphore,
            _SEMAPHORE_TIMEOUT=_SEMAPHORE_TIMEOUT,
            api_key_header=_api_key_header,
            api_secret_key=_API_SECRET_KEY,
            get_real_ip=_get_real_ip,
            check_rate_limit=_check_rate_limit,
            rate_limit_window=_RATE_LIMIT_WINDOW,
            resolve_tables_hybrid=resolve_tables_hybrid,
            max_question_len=_MAX_QUESTION_LEN,
        )
    """
    import secrets as _secrets  # local import — no new dependency

    # ── Store injected values ─────────────────────────────────────────────────
    _INJECTED.update({
        "pipeline":               pipeline,
        "extract_schema_signals": extract_schema_signals,
        "build_lov_context_block": build_lov_context_block,
        "post_process_sql":       post_process_sql,
        "run_llm_async":          run_llm_async,
        "build_prompt":           build_prompt,
        "META":                   META,
        "NARROW_TABLE_INDEX":     NARROW_TABLE_INDEX,
        "_MAX_RESPONSE_ROWS":     _MAX_RESPONSE_ROWS,
        "_DEBUG_RESPONSES":       _DEBUG_RESPONSES,
        "_LLM_MAX_TOKENS_SIMPLE": _LLM_MAX_TOKENS_SIMPLE,
        "_LLM_MAX_TOKENS_COMPLEX": _LLM_MAX_TOKENS_COMPLEX,
        "_LLM_COMPLEX_INTENTS":   _LLM_COMPLEX_INTENTS,
        "_request_semaphore":     _request_semaphore,
        "_SEMAPHORE_TIMEOUT":     _SEMAPHORE_TIMEOUT,
    })

    # ── Resolve security dependency ───────────────────────────────────────────
    _header = api_key_header or APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _verify_key(key: str = Security(_header)) -> None:
        if not _secrets.compare_digest(key or "", api_secret_key):
            raise HTTPException(status_code=403, detail="Forbidden")

    # ── Define the endpoint ───────────────────────────────────────────────────
    @app.post(
        "/query/stream",
        summary="Streaming NL→SQL query (SSE)",
        description=(
            "Same as POST /query but returns a **text/event-stream** response.\n\n"
            "Each `data:` line is a JSON object with an `event` field:\n"
            "- `start` — question received\n"
            "- `tables` — resolved table list\n"
            "- `thinking` — LLM call started\n"
            "- `cache_hit` — full result from cache (terminal)\n"
            "- `sql` — generated SQL\n"
            "- `rows` — query result rows\n"
            "- `done` — stream complete\n"
            "- `error` — pipeline error\n"
        ),
        dependencies=[Depends(_verify_key)],
    )
    async def stream_query_endpoint(
        request: Request,
        req: StreamQueryRequest,
    ) -> StreamingResponse:

        # ── Basic request validation ──────────────────────────────────────────
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")
        if len(req.question) > max_question_len:
            raise HTTPException(
                status_code=400,
                detail=f"question exceeds maximum length of {max_question_len} characters",
            )

        if _INJECTED.get("pipeline") is None:
            raise HTTPException(
                status_code=503, detail="Model is still loading, please retry shortly."
            )

        # ── Rate limiting ─────────────────────────────────────────────────────
        if get_real_ip and check_rate_limit:
            client_ip_rl = get_real_ip(request)
            if not check_rate_limit(client_ip_rl):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please slow down.",
                    headers={"Retry-After": str(rate_limit_window)},
                )

        # ── Table resolution (mirrors /query logic exactly) ───────────────────
        module_key   = (req.module or "").strip()
        subgroup_key = (req.subgroup or "").strip()
        _path_a_hits: Optional[Dict[str, int]] = None

        if req.tables:
            tables = req.tables
            if len(tables) > 1 and module_key and subgroup_key and extract_schema_signals:
                _path_a_hits, _, _ = extract_schema_signals(req.question, tables)
                if not _path_a_hits and resolve_tables_hybrid:
                    _hybrid = resolve_tables_hybrid(
                        question=req.question,
                        module_name=module_key,
                        subgroup_name=subgroup_key,
                    )
                    if _hybrid:
                        tables = _hybrid
                        _path_a_hits = None

        elif module_key and subgroup_key:
            if resolve_tables_hybrid is None:
                raise HTTPException(
                    status_code=500,
                    detail="resolve_tables_hybrid not registered — cannot resolve tables.",
                )
            tables = resolve_tables_hybrid(
                question=req.question,
                module_name=module_key,
                subgroup_name=subgroup_key,
            )
            if not tables:
                raise HTTPException(
                    status_code=400,
                    detail=f"Subgroup '{subgroup_key}' not found under module '{module_key}'.",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide a valid module+subgroup or explicit tables list.",
            )

        # ── Return SSE stream ─────────────────────────────────────────────────
        return StreamingResponse(
            _stream_query(
                question=req.question,
                tables=tables,
                module_key=module_key,
                subgroup_key=subgroup_key,
                precomputed_table_hits=_path_a_hits,
            ),
            media_type="text/event-stream",
            headers={
                # Prevent buffering in Nginx / reverse proxies
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":       "keep-alive",
            },
        )

    log.info("[streaming] POST /query/stream registered.")
