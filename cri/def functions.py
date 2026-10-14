# ============================================================
# 1. ALL IMPORTS - KEEP AS IS
# ============================================================
import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import deque
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# ... ALL your existing imports ...

# ============================================================
# 2. PROMETHEUS METRICS - KEEP AS IS
# ============================================================
_METRICS_ENABLED: bool = False
try:
    from prometheus_client import ...
    # ... all metrics code ...
except ImportError:
    pass

# ============================================================
# 3. RATE LIMITING & QUEUE - KEEP AS IS
# ============================================================
_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
# ... all rate limiting code ...

# ============================================================
# 4. LOGGING - KEEP AS IS
# ============================================================
_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
# ... all logging code ...

# ============================================================
# 5. CONFIG - KEEP AS IS
# ============================================================
_LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").strip().lower()
# ... all config variables ...
# ... MODULE_DATA, META, FULL_SCHEMA_CACHE, etc ...

# ============================================================
# 6. COLUMN ROLE INDEX - KEEP AS IS
# ============================================================
_COL_DESC_ROLE_SIGNALS: Dict[str, List[str]] = { ... }
def _classify_col_desc(description: str) -> List[str]: ...
def _build_col_role_index(module_data: dict) -> Dict[str, Dict[str, List[str]]]: ...
COL_ROLE_INDEX: Dict[str, Dict[str, List[str]]] = _build_col_role_index(MODULE_DATA)

# ============================================================
# 7. ALL HELPER FUNCTIONS - KEEP AS IS
# ============================================================
def _get_table_pk_cols(table: str) -> set: ...
def _get_join_fk_cols(join_clause: Optional[str]) -> set: ...
def _question_mentions_any(question: str, terms: tuple) -> bool: ...
def _schema_audit_cols_for_table(table: str) -> set: ...
def _is_audit_column_for_table(col: str, table: str = "") -> bool: ...
def _is_mandatory_column(...) -> bool: ...
def _build_col_rerank_text(col: str, table: str) -> str: ...
def rerank_columns_cross_encoder(...) -> List[Tuple[str, float]]: ...
def retrieve_relevant_columns(...) -> Dict[str, List[str]]: ...
# ... ALL your existing helper functions ...

# ============================================================
# 8. DATE HELPERS - KEEP AS IS
# ============================================================
def _get_all_date_cols(table: str) -> set: ...
def _col_intent(col_name: str) -> str: ...
def _date_cols_in_where(sql: str, table: str) -> List[str]: ...
def find_date_column(table: str, prefer_creation: bool = True) -> Optional[str]: ...
def inject_effective_filter(sql: str, table: str, question: str) -> str: ...
def inject_time_filter(sql: str, table: str, question: str) -> str: ...

# ============================================================
# 9. SQL UTILITIES - KEEP AS IS
# ============================================================
def _extract_sql_tables(sql: str) -> List[str]: ...
def convert_mysql_to_sqlserver(sql: str) -> str: ...
_PLACEHOLDER_TOKEN_RE = re.compile(r"<\s*[A-Za-z0-9_]+\s*>")
def scrub_invalid_sql_tokens(sql: str) -> str: ...
def _enclosing_select(node) -> Optional["sqlglot.exp.Select"]: ...
def _direct_from_join_tables(select: "sqlglot.exp.Select") -> List["sqlglot.exp.Table"]: ...
def _direct_subquery_aliases(select: "sqlglot.exp.Select") -> set: ...
class _SqlResolutionContext: ...  # KEEP THIS ENTIRE CLASS
def _resolve_columns(sql: str, all_tables: List[str], live_cols: Optional[Dict[str, List[str]]] = None) -> str: ...
def _clean_order_and_group_by(...) -> str: ...
def _fix_group_by_coherence(sql: str) -> str: ...
def _sqlglot_replace_select_star(sql: str, tables: List[str], question: str, intent: str = "simple") -> str: ...
def _uppercase_string_literals(sql: str, all_tables: List[str]) -> str: ...
def _fix_all_string_literals(sql: str, all_tables: List[str]) -> str: ...
def _strip_status_filters_in_ast(expr: "sqlglot.exp.Expression", question: str) -> bool: ...
def _remove_hallucinated_dates_in_ast(expr: "sqlglot.exp.Expression", base_table: str) -> Tuple["sqlglot.exp.Expression", bool]: ...
def _sqlglot_strip_spurious_status_filter(sql: str, question: str) -> str: ...
_STMT_SPLIT_RE = re.compile(r"'(?:[^']|'')*'|(;)")
def _sqlglot_ensure_single_statement(sql: str) -> str: ...
def _ensure_single_statement_regex(sql: str) -> str: ...
def _enforce_join_clause(sql: str, verified_join_clause: str) -> str: ...
_HAS_SQLGLOT_TOP = hasattr(sqlglot.exp, "Top")
def _enforce_top_limit_and_order_in_ast(...) -> Tuple[bool, bool]: ...
def _splice_top_into_sql(sql: str, default_top: int = _DEFAULT_TOP) -> str: ...
def enforce_top_limit(sql: str, default_top: int = _DEFAULT_TOP) -> str: ...
def enforce_default_order(sql: str, tables: List[str], full_schema: dict) -> str: ...
def extract_first_valid_sql(text: str) -> Optional[str]: ...

# ============================================================
# 10. JOIN RESOLUTION FUNCTIONS - KEEP AS IS
# ============================================================
def _find_direct_fk_join(base: str, targets: list, all_tables: Optional[List[str]] = None) -> Optional[str]: ...
def get_neighbor_tables(owner_table: str, hops: int = 2) -> List[str]: ...
def _is_master_table(t: str) -> bool: ...
def _relay_reason(path: List[dict]) -> Optional[str]: ...
def build_join_clause(base: str, tables: list, intent: str = "simple") -> str: ...

# ============================================================
# 11. JOIN SHAPE RESOLUTION - KEEP AS IS
# ============================================================
def _resolve_join_shape(sql: str, base_table: str, intent: str) -> str:
    """
    Rewrites JOINs to N-side (transaction) tables that are used ONLY as a
    filter into correlated EXISTS semi-joins...
    """
    # ... KEEP THIS ENTIRE FUNCTION AS IS ...

# ============================================================
# 12. NEW DISTINCT IMPLEMENTATION - ADD THIS NEW CODE
# ============================================================
def _has_window_functions(expr) -> bool:
    """
    Check if the SQL has window functions.
    """
    window_types = (
        sqlglot.exp.Window,
        sqlglot.exp.RowNumber,
        sqlglot.exp.Rank,
        sqlglot.exp.DenseRank,
        sqlglot.exp.Ntile,
        sqlglot.exp.Lag,
        sqlglot.exp.Lead,
        sqlglot.exp.FirstValue,
        sqlglot.exp.LastValue,
    )
    return any(expr.find(wtype) is not None for wtype in window_types)

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
    """
    def _get_pk_columns(table: str) -> set:
        pk = {c.lower() for c in schema.get(table, {}).get("pk", [])}
        if pk:
            return pk
        for col, info in META.get(table, {}).get("columns", {}).items():
            if isinstance(info, dict) and info.get("is_pk", False):
                pk.add(col.lower())
        for col, roles in COL_ROLE_INDEX.get(table, {}).items():
            if "identifier" in roles:
                pk.add(col.lower())
        return pk
    
    def _get_on_columns(alias: str) -> set:
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
    """
    if not sql or intent in ("relational_aggregation", "aggregation", "group_by"):
        return False
    if len(join_tables) < 2:
        return False
    if "JOIN" not in sql.upper():
        return False
    if re.search(r"\bSELECT\s+DISTINCT\b", sql, re.IGNORECASE):
        return False
    
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if select is None:
            return False
        
        if _has_window_functions(expr):
            log.info("[distinct] Window functions present - skipping DISTINCT")
            return False
        
        joins = list(select.args.get("joins") or [])
        if not joins:
            return False
        
        alias_map: Dict[str, str] = {}
        from_ = select.find(sqlglot.exp.From)
        if from_ is not None and isinstance(from_.this, sqlglot.exp.Table):
            t = from_.this
            alias_map[(t.alias_or_name or t.name).upper()] = (t.name or "").upper()
        for j in joins:
            if isinstance(j.this, sqlglot.exp.Table):
                t = j.this
                alias_map[(t.alias_or_name or t.name).upper()] = (t.name or "").upper()
        
        projected_tables: set = set()
        for scope in ("expressions", "group", "order"):
            for node in select.args.get(scope) or []:
                for col in node.find_all(sqlglot.exp.Column):
                    if col.table:
                        projected_tables.add(col.table.upper())
        
        if not projected_tables:
            for exp in select.expressions:
                if isinstance(exp, sqlglot.exp.Star):
                    log.info("[distinct] SELECT * detected - skipping DISTINCT")
                    return False
        
        many_side_tables: set = set()
        schema = full_schema if full_schema is not None else FULL_SCHEMA_CACHE
        
        for join in joins:
            if not isinstance(join.this, sqlglot.exp.Table):
                continue
            right_alias = (join.this.alias_or_name or join.this.name).upper()
            on_expr = join.args.get("on")
            if on_expr is None:
                continue
            on_tables = {c.table.upper() for c in on_expr.find_all(sqlglot.exp.Column) if c.table}
            left_alias = next((t for t in on_tables if t != right_alias and t in alias_map), None)
            if left_alias is None:
                continue
            left_table = alias_map.get(left_alias, left_alias)
            right_table = alias_map.get(right_alias, right_alias)
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
                many_side_tables.add(right_alias)
                log.info("[distinct] Found 1:N join: %s -> %s", left_alias, right_alias)
        
        if not many_side_tables:
            log.info("[distinct] No 1:N joins found")
            return False
        
        projected_many_side = projected_tables & many_side_tables
        if projected_many_side:
            log.info("[distinct] Many-side columns projected (%s) - skipping DISTINCT", projected_many_side)
            return False
        
        log.info("[distinct] ✅ DISTINCT should be applied")
        return True
        
    except Exception as exc:
        log.warning("[distinct] Error analyzing query: %s - skipping DISTINCT", exc)
        return False

# ============================================================
# 13. POST_PROCESS_SQL - KEEP BUT UPDATE THE JOIN PART
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
    # ... KEEP ALL THE EXISTING CODE ...
    # ... UP TO THE JOIN SHAPE RESOLUTION SECTION ...
    
    # ── 1:N join-shape resolution ──────────────────────────────────────────
    if len(all_tables) > 1:
        sql = _resolve_join_shape(sql, base_table, intent)
        
        # ── NEW: Conservative DISTINCT injection ──────────────────────────
        if _should_apply_distinct(sql, intent, join_tables, FULL_SCHEMA_CACHE):
            if not re.search(r"\bSELECT\s+DISTINCT\b", sql, re.IGNORECASE):
                sql = re.sub(r"\bSELECT\b", "SELECT DISTINCT", sql, count=1, flags=re.IGNORECASE)
                log.info("[post_process] Added DISTINCT (safe 1:N join with parent-only projection)")
                _vprint(
                    "PIPELINE", "green",
                    "🔄 Added DISTINCT - safe 1:N join with parent-only projection",
                    f"tables={join_tables}"
                )
    
    # ... KEEP THE REST OF THE CODE ...
    return sql.strip()

# ============================================================
# 14. ALL REMAINING FUNCTIONS - KEEP AS IS
# ============================================================
def _col_workflow_score(col: str, question: str) -> float: ...
def resolve_lov_matches(...) -> dict: ...
def build_lov_context_block(table: str) -> str: ...
def fix_lov_in_sql(sql: str, table: str) -> str: ...
def _fix_lov_values_consolidated(sql: str, all_tables: List[str]) -> str: ...
def _extract_question_words(question: str) -> List[str]: ...
def extract_schema_signals(...) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]: ...
def _strong_table_hits(table_hits: Dict[str, int]) -> List[str]: ...
def _has_valid_join_path(tables: List[str]) -> Optional[str]: ...
_INTENT_ROLE_PRIORITY: Dict[str, List[str]] = { ... }
_ROLE_BOOST_BY_RANK = [4.0, 3.0, 2.0, 1.0]
def _tokenize(text: str): ...
def _split_column(col: str) -> List[str]: ...
def _score_column(col: str, query_tokens: set) -> int: ...
def _is_audit_column(col: str) -> bool: ...
def _intent_column_boost(col: str, intent: str, table: str = "") -> float: ...
def _auto_select_columns(...) -> List[str]: ...
def _auto_select_columns_multi(...) -> Dict[str, List[str]]: ...
def build_query_aware_lov_context(table: str, question: str, prompt_cols: List[str]) -> str: ...
def _question_implies_join_schema(question: str, tables: List[str]) -> bool: ...
def _should_join(table_scores: Dict[str, int], dominance_factor: float = _JOIN_DOMINANCE_FACTOR) -> bool: ...
def _resolve_join_intent(...) -> str: ...
def _prune_to_needed_tables(...) -> List[str]: ...
def _remove_hallucinated_date_filters(sql: str, table: str) -> str: ...
def get_shared_columns(tables: List[str], live_cols: Dict[str, List[str]]) -> Dict[str, List[str]]: ...
def _ast_fix_column_errors(sql: str, schema_errors: List[str]) -> Optional[str]: ...
def validate_sql(sql: str, selected_tables: List[str]) -> List[str]: ...
def enforce_mandatory_select_columns(...) -> str: ...

# ============================================================
# 15. SEMANTIC CACHE - KEEP AS IS
# ============================================================
class SemanticCache:
    # ... KEEP ENTIRE CLASS ...

# ============================================================
# 16. LLM CIRCUIT BREAKER - KEEP AS IS
# ============================================================
_LLM_CB_FAILURE_THRESHOLD: int = int(os.getenv("LLM_CB_FAILURE_THRESHOLD", "5"))
# ... all circuit breaker code ...
class LLMCircuitBreaker: ...
_llm_circuit_breaker = LLMCircuitBreaker()
async def _call_llm_server(prompt: str, max_tokens: int) -> Tuple[Optional[str], Optional[str], bool]: ...
async def run_llm_async(prompt: str, max_tokens: int = 320) -> Tuple[Optional[str], Optional[str]]: ...

# ============================================================
# 17. NL2SQL PIPELINE CLASS - KEEP AS IS
# ============================================================
class NL2SQLPipeline:
    # ... KEEP ENTIRE CLASS ...

# ============================================================
# 18. REVERSE LOOKUP - KEEP AS IS
# ============================================================
_MASTER_CACHE_PATH = Path(os.getenv("MASTER_TRANSLATION_CACHE_PATH", str(_PROJECT_ROOT / "data" / "master_translation_cache.json")))
_REVERSE_LOOKUP_INDEX: Dict[str, List[dict]] = {}
_reverse_lookup_lock = _threading.Lock()
def _build_reverse_lookup_index(cache_path: Path = _MASTER_CACHE_PATH) -> Dict[str, List[dict]]: ...
def reload_reverse_lookup_index() -> None: ...
def _reverse_lookup_question_values(question: str) -> List[dict]: ...
def _fix_reverse_lookup_in_sql(sql: str, reverse_hits: List[dict]) -> str: ...
def _fix_reverse_lookup_subqueries(sql: str, pre_injected_filters: List[dict]) -> str: ...
def _fix_numeric_col_string_literal_in_sql(sql: str, all_tables: List[str]) -> str: ...

# ============================================================
# 19. FASTAPI APP - KEEP AS IS
# ============================================================
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
_global_embedder: Optional[SentenceTransformer] = None
_shared_qdrant_client: Optional[QdrantClient] = None
_audit_executor = ThreadPoolExecutor(max_workers=2)
def log_request_to_db(...): ...
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... KEEP ENTIRE FUNCTION ...
pipeline: Optional[NL2SQLPipeline] = None
class QueryRequest(BaseModel): ...
class QueryResponse(BaseModel): ...
class RunSqlRequest(BaseModel): ...
app = FastAPI(title="NL2SQL MES Chatbot", version="3.0", lifespan=lifespan)
# ... ALL ROUTES AND ENDPOINTS ...

# ============================================================
# 20. FINAL APP RUN - KEEP AS IS
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)