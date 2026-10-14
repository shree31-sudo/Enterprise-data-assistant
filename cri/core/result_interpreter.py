from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

log = logging.getLogger("NL2SQL")

# ── Config ─────────────────────────────────────────────────────────────────────
_ENABLED: bool = (
    os.getenv("RESULT_INTERPRETER_ENABLED", "true").strip().lower() in ("1", "true", "yes")
)

_AGENT_LLM_URL: str = os.getenv(
    "AGENT_LLM_URL",
    os.getenv("LLM_SERVER_URL", "http://localhost:8081/completion"),
).strip()

# "llamacpp" -> POST {prompt, n_predict, ...} to /completion, reads data["content"]
# "ollama"   -> POST {model, prompt, raw:true, options:{...}} to /api/generate, reads data["response"]
_AGENT_LLM_BACKEND: str = os.getenv("AGENT_LLM_BACKEND", "llamacpp").strip().lower()
_AGENT_OLLAMA_MODEL: str = os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5-3b-instruct").strip()

_TIMEOUT: float          = float(os.getenv("AGENT_LLM_TIMEOUT_SECS",    "45").strip())
_PRE_CALL_DELAY: float   = float(os.getenv("AGENT_PRE_CALL_DELAY_SECS", "1.0").strip())
_MAX_TOKENS: int         = int(os.getenv("AGENT_LLM_MAX_TOKENS",        "150").strip())
_MAX_ROWS_IN_PROMPT: int = int(os.getenv("AGENT_MAX_ROWS_IN_PROMPT",    "10").strip())

_AGENT_STOP: list = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]

# ── Output dataclass ───────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    summary:   str
    narrative: Optional[str] = None
    insight:   Optional[str] = None   
    followup:  Optional[str] = None  
    reasoning: Optional[str] = None
    status:    str           = "ok"  
    turns:     int           = 1


# ── Deterministic tools — all run in Python, no LLM needed ────────────────────

def _tool_classify_result(rows: List[Dict], row_count: int, intent: str) -> tuple[str, str]:
    """Returns (status, classification_text)."""
    if row_count == 0:
        return "zero", "Query returned no rows."
    if row_count == 1 and intent in ("aggregation", "group_by", "existence"):
        val = list(rows[0].values())[0] if rows[0] else None
        if val in (0, "0", None, ""):
            return "suspect", "Single aggregation row with zero/null value — may indicate a filter issue."
        return "ok", f"Single aggregation row — value={val}."
    col_names = list(rows[0].keys()) if rows else []
    id_only = all(c.upper().endswith("_ID") or c.upper() == "ID" for c in col_names)
    if id_only and row_count > 0:
        return "suspect", "Result contains only ID columns — likely missing a JOIN for display names."
    return "ok", f"{row_count} row(s) returned with columns: {', '.join(col_names)}."


def _tool_summarise_rows(rows: List[Dict], row_count: int, truncated: bool) -> str:
    """Extract key readable values from rows."""
    sample = rows[:_MAX_ROWS_IN_PROMPT]
    if not sample:
        return "No rows to summarise."
    col_names = list(sample[0].keys())
    label_cols = [
        c for c in col_names
        if not c.upper().endswith("_ID") and "FLAG" not in c.upper()
    ][:4] or col_names[:4]
    summaries = []
    for r in sample[:5]:
        parts = [f"{c}={r.get(c, '')}" for c in label_cols]
        summaries.append(", ".join(parts))
    trunc = f" (first {len(sample)} of {row_count} shown)" if truncated or row_count > len(sample) else ""
    return (
        f"Total rows: {row_count}{trunc}\n"
        f"Columns: {', '.join(label_cols)}\n"
        f"Values:\n" + "\n".join(f"  • {s}" for s in summaries)
    )


def _tool_diagnose_zero(question: str, intent: str) -> str:
    """Keyword-based MES cause analysis for zero-row results."""
    causes = []
    q_lower = question.lower()
    if any(k in q_lower for k in ("today", "now", "current")):
        causes.append("date filter may be too tight — try widening to this week or month")
    if any(k in q_lower for k in ("active", "open", "pending")):
        causes.append("status filter may be hiding records — try querying all statuses")
    if any(k in q_lower for k in ("work order", "bom", "routing", "item", "resource")):
        causes.append("entity name or code mismatch — verify the exact name/code in the system")
    if intent in ("relational_lookup", "join"):
        causes.append("missing JOIN data — the linked record may not exist yet")
    causes.append("data may not have been entered into the MES system yet")
    return "Likely causes: " + "; ".join(causes) + "."


def _tool_diagnose_suspect(rows: List[Dict], row_count: int) -> str:
    """Diagnose why data looks incomplete."""
    col_names = list(rows[0].keys()) if rows else []
    issues = []
    id_cols   = [c for c in col_names if c.upper().endswith("_ID")]
    name_cols = [c for c in col_names if "NAME" in c.upper() or "DESC" in c.upper() or "CODE" in c.upper()]
    if id_cols and not name_cols:
        issues.append(f"has ID columns ({', '.join(id_cols)}) but no display name columns — JOIN may be missing")
    null_cols = [c for c in col_names if all(r.get(c) in (None, "", "None", "null") for r in rows[:5])]
    if null_cols:
        issues.append(f"columns with all-null values: {', '.join(null_cols)}")
    if row_count == 1:
        issues.append("only one row returned — filter may be too specific")
    if not issues:
        issues.append("data structure looks unusual — verify the correct table is being queried")
    return "Issue: result " + "; ".join(issues) + "."


# ── Context builder — runs all tools, produces pre-computed observations ───────

def _build_context(
    question: str,
    rows: List[Dict],
    row_count: int,
    intent: str,
    truncated: bool,
) -> tuple[str, str]:
    """
    Runs all deterministic tools and returns (status, context_text).
    context_text is fed directly to the LLM — no LLM needed for analysis.
    """
    status, classification = _tool_classify_result(rows, row_count, intent)
    summary_text           = _tool_summarise_rows(rows, row_count, truncated)

    parts = [
        f"Classification: {classification}",
        f"Data:\n{summary_text}",
    ]

    if status == "zero":
        parts.append(_tool_diagnose_zero(question, intent))
    elif status == "suspect":
        parts.append(_tool_diagnose_suspect(rows, row_count))

    return status, "\n\n".join(parts)


# ── Prompt builder — LLM is writer only, all thinking already done ─────────────

def _build_prompt(
    question: str,
    row_count: int,
    status: str,
    context: str,
) -> str:
    if status == "zero":
        task = (
            "No records were found. In 1-2 sentences tell the user what was searched "
            "and the most likely fix from the diagnosis. Be direct and helpful."
        )
    elif status == "suspect":
        task = (
            "The result looks incomplete. In 1-2 sentences describe what was returned "
            "and what seems wrong, using the diagnosis to explain the likely cause."
        )
    else:
        task = (
            "Write 2-3 sentences directly answering the question. "
            "Use the ACTUAL values from the data — specific names, numbers, codes, "
            "or statuses. Do NOT say 'records were found' or 'the query returned'. "
            "Write as if you know the data personally."
        )

    return (
        "<|im_start|>system\n"
        "You are a MES assistant. Answer the user's question using the data provided.\n"
        "Use real data values. Do NOT mention SQL or databases. Be concise.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {question}\n"
        f"Row count: {row_count}\n\n"
        f"{context}\n\n"
        f"Task: {task}\n"
        "Output JSON only: {\"narrative\": \"...\"}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{"
    )


# ── LLM caller ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str, pre_delay: float = 0.0) -> Optional[str]:
    if pre_delay > 0:
        await asyncio.sleep(pre_delay)

    try:
        async with aiohttp.ClientSession() as session:
            if _AGENT_LLM_BACKEND == "ollama":
                payload: Dict[str, Any] = {
                    "model":  _AGENT_OLLAMA_MODEL,
                    "prompt": prompt,
                    "raw":    True,  # bypass Ollama's chat templating — use prompt verbatim
                    "stream": False,
                    "options": {
                        "num_predict":    _MAX_TOKENS,
                        "temperature":    0.1,
                        "repeat_penalty": 1.05,
                        "stop":           _AGENT_STOP,
                    },
                }
            else:
                payload = {
                    "prompt":         prompt,
                    "n_predict":      _MAX_TOKENS,
                    "temperature":    0.1,
                    "repeat_penalty": 1.05,
                    "stop":           _AGENT_STOP,
                    "stream":         False,
                }
            async with session.post(
                _AGENT_LLM_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    log.warning("[agent] LLM HTTP %d", resp.status)
                    return None
                data = await resp.json()
                if _AGENT_LLM_BACKEND == "ollama":
                    raw = (data.get("response") or "").strip()
                else:
                    raw = (data.get("content") or "").strip()
                log.debug("[agent] raw output: %.300s", raw)
                return raw if raw else None

    except asyncio.TimeoutError:
        log.warning("[agent] LLM timeout after %.0fs", _TIMEOUT)
        return None
    except Exception as exc:
        log.warning("[agent] call failed: %s", exc)
        return None


# ── JSON parser ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    if not raw.startswith("{"):
        raw = "{" + raw
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", m.group())
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# ── Fallback ───────────────────────────────────────────────────────────────────

def _fallback(row_count: int, status: str = "ok") -> AgentResult:
    if row_count == 0:
        msg = "No records were found for that query."
        return AgentResult(summary=msg, narrative=msg, status="zero", turns=0)
    msg = f"Found {row_count} record{'s' if row_count != 1 else ''} matching your query."
    return AgentResult(summary=msg, narrative=msg, status=status, turns=0)


# ── Agent entry point — single LLM call ───────────────────────────────────────

async def interpret_result(
    question: str,
    rows: List[Dict[str, Any]],
    row_count: int,
    intent: str     = "simple",
    truncated: bool = False,
) -> AgentResult:
    """
    Single-turn interpreter:
      1. All deterministic tools run in Python (zero latency)
      2. One LLM call writes the narrative using pre-computed context
    Always returns AgentResult — never raises.
    """
    if not _ENABLED:
        return _fallback(row_count)

    t_start = time.perf_counter()

    # ── Step 1: run all tools deterministically (no LLM) ──────────────────
    status, context = _build_context(question, rows, row_count, intent, truncated)

    log.info(
        "[agent] START q=%.60r rows=%d intent=%s status=%s",
        question, row_count, intent, status,
    )

    # ── Step 2: single LLM call — write narrative only ────────────────────
    prompt = _build_prompt(question, row_count, status, context)
    raw    = await _call_llm(prompt, pre_delay=_PRE_CALL_DELAY)

    if not raw:
        log.warning("[agent] LLM returned nothing — using deterministic fallback")
        return _fallback(row_count, status)

    data = _parse_json(raw)

    if not data:
        log.warning("[agent] JSON parse failed — raw=%.120s", raw)
        return _fallback(row_count, status)

    # ── Step 3: assemble result ────────────────────────────────────────────
    narrative  = (data.get("narrative") or "").strip()
    llm_status = (data.get("status") or status).lower()

    if llm_status not in ("ok", "zero", "suspect"):
        llm_status = status  

    if not narrative:
        log.warning("[agent] empty narrative — using fallback")
        return _fallback(row_count, llm_status)

    summary = narrative.split(".")[0].strip() or narrative[:80]

    log.info(
        "[agent] ✅ done — status=%s elapsed=%.2fs",
        llm_status, time.perf_counter() - t_start,
    )

    return AgentResult(
        summary   = summary,
        narrative = narrative,
        insight   = None,
        followup  = None,
        reasoning = context[:300],
        status    = llm_status,
        turns     = 1,
    )
