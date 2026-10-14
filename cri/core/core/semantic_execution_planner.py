from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger("NL2SQL")

# ─────────────────────────────────────────────────────────────────────────────
# Tunable weights (override via environment variables)
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS: Dict[str, float] = {
    "semantic":    float(os.getenv("SEP_W_SEMANTIC",    "0.40")),
    "column":      float(os.getenv("SEP_W_COLUMN",      "0.25")),
    "temporal":    float(os.getenv("SEP_W_TEMPORAL",    "0.15")),
    "aggregation": float(os.getenv("SEP_W_AGGREGATION", "0.10")),
    "hybrid":      float(os.getenv("SEP_W_HYBRID",      "0.10")),
}

# Minimum score for a table to be considered a semantic execution candidate
_MIN_SCORE_THRESHOLD: float = float(os.getenv("SEP_MIN_SCORE", "0.20"))

# If top-scoring table beats second table by this margin, keep only the top one
# (prevents unnecessary joins)
_DOMINANCE_MARGIN: float = float(os.getenv("SEP_DOMINANCE_MARGIN", "0.35"))

# Maximum execution tables returned (before graph resolver adds bridge tables)
_MAX_EXECUTION_TABLES: int = int(os.getenv("SEP_MAX_TABLES", "3"))

# ─────────────────────────────────────────────────────────────────────────────
# Module prefix stripping — read from env so new modules need zero code changes
# ─────────────────────────────────────────────────────────────────────────────

def _build_strip_prefixes() -> tuple:
    """
    Build a tuple of module prefixes to strip from table names when
    humanising them for BM25/embedding corpus text.

    Sources (first non-empty wins):
      1. TABLE_NAME_PREFIXES   e.g. "MFC_M_,MFC_L_,MFC_T_,ALM_M_,ALM_L_"
      2. TABLE_PREFIX          e.g. "MFC,ALM,USR"
         → expanded to "<CODE>_M_", "<CODE>_L_", "<CODE>_T_", "<CODE>_"
      3. Hard fallback         → ("MFC_M_", "MFC_L_", "MFC_T_", "MFC_")
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


_STRIP_PREFIXES: tuple = _build_strip_prefixes()


def _strip_table_prefix(name: str) -> str:
    """
    Strip any known module prefix from a table name.
    Tries longest prefix first to avoid partial matches
    (e.g. "MFC_" must not match before "MFC_M_").
    """
    for pfx in sorted(_STRIP_PREFIXES, key=len, reverse=True):
        if name.upper().startswith(pfx.upper()):
            return name[len(pfx):]
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Semantic role signals — used to detect what kind of info the query needs
# ─────────────────────────────────────────────────────────────────────────────
_TEMPORAL_SIGNALS: Set[str] = {
    "date", "time", "when", "month", "year", "day", "week", "quarter",
    "created", "modified", "updated", "since", "between", "during",
    "before", "after", "period", "from", "till", "until", "today",
    "yesterday", "recent", "latest", "scheduled", "expir", "effective",
    "start", "end", "deadline", "due",
}

_AGGREGATION_SIGNALS: Set[str] = {
    "count", "total", "sum", "average", "avg", "max", "min", "how many",
    "number of", "quantity", "amount", "most", "least", "highest",
    "lowest", "aggregate", "per", "group", "breakdown", "distribution",
}

_IDENTITY_SIGNALS: Set[str] = {
    "name", "code", "id", "identifier", "number", "no", "ref",
    "reference", "list", "show", "find", "get", "what", "which",
}

_CLASSIFICATION_SIGNALS: Set[str] = {
    "type", "category", "kind", "class", "classification", "status",
    "state", "flag", "level", "tier", "group", "family",
}

_STATUS_SIGNALS: Set[str] = {
    "active", "inactive", "pending", "completed", "open", "closed",
    "approved", "rejected", "cancelled", "in progress", "done",
    "enabled", "disabled",
}

# ─────────────────────────────────────────────────────────────────────────────
# Query stopwords — tokens that carry no semantic requirement
# ─────────────────────────────────────────────────────────────────────────────
_QUERY_STOPWORDS: Set[str] = {
    "select", "from", "where", "show", "list", "get", "find", "give",
    "tell", "what", "which", "how", "many", "all", "the", "for", "and",
    "with", "its", "their", "that", "have", "has", "are", "was", "been",
    "per", "order", "group", "count", "sum", "avg", "top", "desc", "asc",
    "me", "us", "by", "of", "in", "on", "at", "to", "is", "an", "a",
    "or", "not", "any", "each", "every", "this", "those", "these",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Semantic Requirement Extraction
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequirements:
    """Structured semantic requirements extracted from a natural-language query."""

    def __init__(self) -> None:
        self.concepts: List[str]        = []   # noun-phrase tokens from query
        self.needs_temporal: bool       = False
        self.needs_aggregation: bool    = False
        self.needs_identity: bool       = False
        self.needs_classification: bool = False
        self.needs_status: bool         = False
        self.intent: str                = "simple"

    def __repr__(self) -> str:
        flags = []
        if self.needs_temporal:      flags.append("temporal")
        if self.needs_aggregation:   flags.append("aggregation")
        if self.needs_classification: flags.append("classification")
        if self.needs_status:        flags.append("status")
        if self.needs_identity:      flags.append("identity")
        return (
            f"QueryRequirements(concepts={self.concepts!r}, "
            f"flags={flags}, intent={self.intent!r})"
        )


def extract_query_requirements(question: str, intent: str = "simple") -> QueryRequirements:
    """
    Extract semantic requirements from a natural-language question.

    Returns a QueryRequirements object describing what information the query
    needs — independent of any schema knowledge.

    Parameters
    ----------
    question : str
        The raw NL question.
    intent : str
        Pre-classified intent (from LLM or classifier).  Used as a hint only.
    """
    req = QueryRequirements()
    req.intent = intent

    ql = question.lower()

    # ── Noun-phrase concept extraction ───────────────────────────────────────
    # Tokenise on word boundaries, drop stopwords and very short tokens.
    # Use bigrams too — "bom type", "work order", "created date" etc.
    tokens = re.findall(r"[a-z][a-z0-9_]{1,}", ql)
    meaningful = [t for t in tokens if t not in _QUERY_STOPWORDS and len(t) >= 3]

    # Unigrams
    concepts: List[str] = list(dict.fromkeys(meaningful))

    # Bigrams (sliding window of 2)
    bigrams = [
        f"{tokens[i]} {tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] not in _QUERY_STOPWORDS or tokens[i+1] not in _QUERY_STOPWORDS
    ]
    concepts = list(dict.fromkeys(concepts + bigrams))

    req.concepts = concepts

    # ── Semantic flag detection ───────────────────────────────────────────────
    req.needs_temporal = any(sig in ql for sig in _TEMPORAL_SIGNALS)
    req.needs_aggregation = (
        any(sig in ql for sig in _AGGREGATION_SIGNALS)
        or intent in {"aggregation", "group_by", "top_bottom",
                      "relational_aggregation", "sort_by"}
    )
    req.needs_classification = any(sig in ql for sig in _CLASSIFICATION_SIGNALS)
    req.needs_status = any(sig in ql for sig in _STATUS_SIGNALS)
    req.needs_identity = any(sig in ql for sig in _IDENTITY_SIGNALS)

    log.debug(
        "[SEP] Requirements extracted: %r",
        req,
        extra={"event": "sep_requirements", "question": question[:80]},
    )
    return req


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Table Semantic Capability Representation
# ══════════════════════════════════════════════════════════════════════════════

def build_table_capability_text(table_name: str, meta: dict, role_index: dict) -> str:
    """
    Build a rich semantic capability string for a table from its metadata.

    This is the text we embed or keyword-match against query requirements.
    Sources (in priority order):
      1. business_purpose   — what the table is FOR
      2. row_represents     — what each row means
      3. synonyms / use_when — alternative descriptions
      4. column names       — flattened and humanised
      5. query_patterns     — example questions this table answers
      6. column roles       — from COL_ROLE_INDEX
    """
    tinfo = meta.get(table_name, {})
    parts: List[str] = []

    # Humanise table name — strip any known module prefix (env-driven)
    human_name = _strip_table_prefix(table_name).replace("_", " ").lower()
    parts.append(human_name)

    for field in ("business_purpose", "row_represents"):
        v = tinfo.get(field, "")
        if v:
            parts.append(v.lower())

    for v in tinfo.get("synonyms", []):
        parts.append(v.lower())

    for v in tinfo.get("use_when", []):
        parts.append(v.lower())

    for v in tinfo.get("query_patterns", []):
        parts.append(v.lower())

    # Column names → humanised tokens
    col_tokens: List[str] = []
    for col in tinfo.get("columns", {}).keys():
        col_tokens.extend(re.split(r"[_\s]+", col.lower()))
    if col_tokens:
        parts.append(" ".join(dict.fromkeys(col_tokens)))

    # Column roles (from role_index)
    tbl_roles: Set[str] = set()
    for roles in role_index.get(table_name, {}).values():
        tbl_roles.update(roles)
    if tbl_roles:
        parts.append(" ".join(sorted(tbl_roles)))

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Per-table Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _requirement_coverage_score(
    requirements: QueryRequirements,
    capability_text: str,
    embedder=None,
    query_embedding=None,
    table_embedding=None,
) -> float:
    """
    Measure how well a table's capability text covers the query's semantic
    requirements.

    Strategy:
      - Concept overlap: what fraction of query concepts appear in capability text
      - If embedder available: cosine similarity between query and table vectors
      - Combine both signals (embedding dominates when available)
    """
    score = 0.0

    # ── Concept overlap (keyword-level) ──────────────────────────────────────
    cap_lower = capability_text.lower()
    if requirements.concepts:
        matched = sum(1 for c in requirements.concepts if c in cap_lower)
        overlap = matched / len(requirements.concepts)
    else:
        overlap = 0.0

    # ── Embedding similarity ──────────────────────────────────────────────────
    if query_embedding is not None and table_embedding is not None:
        q = np.array(query_embedding)
        t = np.array(table_embedding)
        # Both should already be normalised (SentenceTransformer normalize=True)
        cos_sim = float(np.dot(q, t))
        # Blend: 60 % embedding + 40 % keyword
        score = 0.60 * cos_sim + 0.40 * overlap
    else:
        score = overlap

    return max(0.0, min(1.0, score))


def _column_capability_score(
    requirements: QueryRequirements,
    table_name: str,
    meta: dict,
    role_index: dict,
) -> float:
    """
    Score how well this table's column roles match the query's semantic flags.
    Uses COL_ROLE_INDEX — no hardcoded column names.
    """
    score = 0.0
    total_checks = 0
    hits = 0

    tbl_roles: Set[str] = set()
    for roles in role_index.get(table_name, {}).values():
        tbl_roles.update(roles)

    check_map = [
        (requirements.needs_temporal,       {"date"}),
        (requirements.needs_aggregation,    {"quantity"}),
        (requirements.needs_classification, {"status", "foreign_key"}),
        (requirements.needs_identity,       {"name", "identifier"}),
        (requirements.needs_status,         {"status"}),
    ]
    for flag, role_set in check_map:
        if flag:
            total_checks += 1
            if tbl_roles & role_set:
                hits += 1

    if total_checks == 0:
        return 0.5  # neutral — no specific column requirements

    return hits / total_checks


def _temporal_alignment_score(
    requirements: QueryRequirements,
    table_name: str,
    role_index: dict,
) -> float:
    """1.0 if query needs temporal info and table has date columns, else 0."""
    if not requirements.needs_temporal:
        return 0.5  # neutral

    date_cols = [
        col for col, roles in role_index.get(table_name, {}).items()
        if "date" in roles
    ]
    return 1.0 if date_cols else 0.0


def _aggregation_alignment_score(
    requirements: QueryRequirements,
    table_name: str,
    role_index: dict,
) -> float:
    """1.0 if query needs aggregation and table has quantity/numeric columns."""
    if not requirements.needs_aggregation:
        return 0.5  # neutral

    numeric_cols = [
        col for col, roles in role_index.get(table_name, {}).items()
        if "quantity" in roles
    ]
    return 1.0 if numeric_cols else 0.0


def _hybrid_position_score(table_name: str, candidate_tables: List[str]) -> float:
    """
    Normalised rank from hybrid retrieval (BM25+embeddings).
    Index 0 = highest BM25/embedding confidence → score 1.0.
    """
    if not candidate_tables:
        return 0.0
    try:
        idx = candidate_tables.index(table_name)
        return 1.0 - (idx / len(candidate_tables))
    except ValueError:
        return 0.0


def score_table(
    table_name: str,
    requirements: QueryRequirements,
    candidate_tables: List[str],
    meta: dict,
    role_index: dict,
    capability_text: str,
    embedder=None,
    query_embedding=None,
    table_embedding=None,
) -> float:
    """
    Composite semantic alignment score for a single candidate table.

    score(t) =
        w_sem  * requirement_coverage(t)
      + w_col  * column_capability(t)
      + w_temp * temporal_alignment(t)
      + w_agg  * aggregation_alignment(t)
      + w_hyb  * hybrid_position(t)
    """
    sem  = _requirement_coverage_score(
               requirements, capability_text,
               embedder, query_embedding, table_embedding)
    col  = _column_capability_score(requirements, table_name, meta, role_index)
    temp = _temporal_alignment_score(requirements, table_name, role_index)
    agg  = _aggregation_alignment_score(requirements, table_name, role_index)
    hyb  = _hybrid_position_score(table_name, candidate_tables)

    total = (
        _WEIGHTS["semantic"]    * sem  +
        _WEIGHTS["column"]      * col  +
        _WEIGHTS["temporal"]    * temp +
        _WEIGHTS["aggregation"] * agg  +
        _WEIGHTS["hybrid"]      * hyb
    )

    log.debug(
        "[SEP] score table=%r  sem=%.3f col=%.3f temp=%.3f agg=%.3f hyb=%.3f → total=%.3f",
        table_name, sem, col, temp, agg, hyb, total,
        extra={"event": "sep_table_score", "table": table_name, "score": total},
    )
    return total


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def plan_execution_tables(
    question: str,
    candidate_tables: List[str],
    intent: str,
    meta: dict,
    role_index: dict,
    embedder=None,
    table_embeddings: Optional[Dict[str, List[float]]] = None,
) -> List[str]:
    """
    Determine which subset of candidate_tables should actually be used for
    SQL execution, based on semantic alignment with the query.

    Parameters
    ----------
    question         : NL question string
    candidate_tables : Ordered list from hybrid retrieval (best first)
    intent           : Query intent classification string
    meta             : META dict (mfc.json keyed by table_name)
    role_index       : COL_ROLE_INDEX from chatbot_backend
    embedder         : SentenceTransformer instance (or None for keyword-only)
    table_embeddings : Pre-built table embedding vectors (or None)

    Returns
    -------
    List[str]
        Execution tables, ordered by semantic alignment score (best first).
        Always contains at least one table.
        Subsequent join resolution should use graph resolver on this list.
    """
    if not candidate_tables:
        log.warning("[SEP] plan_execution_tables called with empty candidate_tables")
        return []

    if len(candidate_tables) == 1:
        log.info("[SEP] Single candidate table — returning as-is: %r", candidate_tables[0])
        return candidate_tables

    # ── Step 1: Extract query semantic requirements ───────────────────────────
    requirements = extract_query_requirements(question, intent)
    log.info(
        "[SEP] STEP 1 — Requirements: %r",
        requirements,
        extra={"event": "sep_step1", "requirements": str(requirements)},
    )

    # ── Step 2: Embed query (if embedder available) ───────────────────────────
    query_embedding = None
    if embedder is not None:
        try:
            query_embedding = embedder.encode(
                question, normalize_embeddings=True
            ).tolist()
        except Exception as e:
            log.warning("[SEP] Failed to embed query: %s", e)

    # ── Step 3: Score each candidate table ───────────────────────────────────
    scored: List[Tuple[str, float]] = []
    for table in candidate_tables:
        cap_text = build_table_capability_text(table, meta, role_index)
        t_emb = (table_embeddings or {}).get(table)
        s = score_table(
            table_name=table,
            requirements=requirements,
            candidate_tables=candidate_tables,
            meta=meta,
            role_index=role_index,
            capability_text=cap_text,
            embedder=embedder,
            query_embedding=query_embedding,
            table_embedding=t_emb,
        )
        scored.append((table, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    log.info(
        "[SEP] STEP 3 — Scores: %s",
        [(t, f"{s:.3f}") for t, s in scored],
        extra={"event": "sep_step3_scores", "scores": {t: round(s, 4) for t, s in scored}},
    )

    # ── Step 4: Determine execution tables ───────────────────────────────────
    # Rule 1: Remove tables below minimum threshold
    qualified = [(t, s) for t, s in scored if s >= _MIN_SCORE_THRESHOLD]
    if not qualified:
        # Every table scored below threshold — fall back to top-ranked hybrid
        log.info(
            "[SEP] STEP 4 — All tables below threshold (%.2f); "
            "falling back to hybrid top: %r",
            _MIN_SCORE_THRESHOLD, candidate_tables[0],
        )
        return [candidate_tables[0]]

    # Rule 2: If top table dominates second by DOMINANCE_MARGIN → single table
    if len(qualified) >= 2:
        top_score   = qualified[0][1]
        second_score = qualified[1][1]
        if top_score - second_score >= _DOMINANCE_MARGIN:
            log.info(
                "[SEP] STEP 4 — Dominant table: %r (score=%.3f, gap=%.3f ≥ %.2f) "
                "→ single execution table, no join needed",
                qualified[0][0], top_score,
                top_score - second_score, _DOMINANCE_MARGIN,
            )
            return [qualified[0][0]]

    # Rule 3: Respect intent — pure temporal/aggregation queries rarely need joins
    if intent in {"time_based", "sort_by"} and not requirements.needs_classification:
        log.info(
            "[SEP] STEP 4 — Intent=%r without classification need → single table: %r",
            intent, qualified[0][0],
        )
        return [qualified[0][0]]

    # Rule 4: Return up to MAX_EXECUTION_TABLES (join resolution happens after)
    execution_tables = [t for t, _ in qualified[:_MAX_EXECUTION_TABLES]]

    log.info(
        "[SEP] STEP 4 — Execution tables selected: %s  (from %d candidates)",
        execution_tables, len(candidate_tables),
        extra={
            "event": "sep_execution_tables",
            "execution_tables": execution_tables,
            "candidate_count": len(candidate_tables),
        },
    )

    return execution_tables


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPER (used in /health endpoint and debug logs)
# ══════════════════════════════════════════════════════════════════════════════

def explain_planning(
    question: str,
    candidate_tables: List[str],
    intent: str,
    meta: dict,
    role_index: dict,
    embedder=None,
    table_embeddings: Optional[Dict[str, List[float]]] = None,
) -> dict:
    """
    Return a human-readable breakdown of the planning decision.
    Safe to call in debug / health-check contexts.
    """
    requirements = extract_query_requirements(question, intent)
    query_embedding = None
    if embedder is not None:
        try:
            query_embedding = embedder.encode(question, normalize_embeddings=True).tolist()
        except Exception:
            pass

    details = []
    for table in candidate_tables:
        cap_text = build_table_capability_text(table, meta, role_index)
        t_emb = (table_embeddings or {}).get(table)
        s = score_table(
            table_name=table,
            requirements=requirements,
            candidate_tables=candidate_tables,
            meta=meta,
            role_index=role_index,
            capability_text=cap_text,
            embedder=embedder,
            query_embedding=query_embedding,
            table_embedding=t_emb,
        )
        details.append({
            "table": table,
            "score": round(s, 4),
            "capability_text_preview": cap_text[:120],
        })
    details.sort(key=lambda x: x["score"], reverse=True)

    execution = plan_execution_tables(
        question, candidate_tables, intent, meta, role_index,
        embedder, table_embeddings,
    )

    return {
        "question": question,
        "intent": intent,
        "requirements": {
            "concepts": requirements.concepts[:10],
            "needs_temporal": requirements.needs_temporal,
            "needs_aggregation": requirements.needs_aggregation,
            "needs_classification": requirements.needs_classification,
            "needs_status": requirements.needs_status,
            "needs_identity": requirements.needs_identity,
        },
        "table_scores": details,
        "execution_tables": execution,
        "join_required": len(execution) > 1,
    }