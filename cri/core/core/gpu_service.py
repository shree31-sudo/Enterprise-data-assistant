import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlglot
import sqlglot.errors as _sqlglot_errors
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("GPU_SERVICE")

# ── Environment ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_ROOT.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

# GPU Service Config
_GPU_HOST: str = os.getenv("GPU_HOST", "0.0.0.0")
_GPU_PORT: int = int(os.getenv("GPU_PORT", "8103"))

# Ollama Config 
_OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://10.225.30.26:11434/api/generate")
_OLLAMA_TAGS_URL: str = _OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "xiyansql7b:latest")
_OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT_SECS", "180"))
_OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "400"))

# Data paths (
_DATA_PATH = Path(os.getenv("GPU_DATA_PATH", str(_REPO_ROOT / "data")))

# ── Queue Config ─────────────────────────────────────────────────────────────
_GPU_QUEUE_WORKERS: int = int(os.getenv("GPU_QUEUE_WORKERS", "1"))
_GPU_QUEUE_MAXSIZE: int = int(os.getenv("GPU_QUEUE_MAXSIZE", "20"))
_GPU_QUEUE_WAIT_TIMEOUT: float = float(os.getenv("GPU_QUEUE_WAIT_TIMEOUT_SECS", "120"))

# ── SQL Validation Helpers ──────────────────────────────────────────────────
def _enclosing_select(node) -> Optional["sqlglot.exp.Select"]:
    """Nearest ancestor SELECT of `node` (not counting `node` itself)."""
    p = node.parent
    while p is not None and not isinstance(p, sqlglot.exp.Select):
        p = p.parent
    return p
_AGGREGATE_FUNCTION_NAMES = {
    "sum", "count", "count_big", "avg", "min", "max",
    "string_agg", "stdev", "stdevp", "var", "varp",
    "grouping", "grouping_id", "checksum_agg", "approx_count_distinct",
}
_AGG_NATIVE_TYPES = (
    sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg,
    sqlglot.exp.Min, sqlglot.exp.Max,
)

def _contains_aggregate(node) -> bool:
    """
    True if `node` contains a real aggregate function call — either one of
    sqlglot's native aggregate node types, or an Anonymous() function call
    whose name matches a known T-SQL aggregate. Replaces the old pattern of
    blanket-matching `sqlglot.exp.Anonymous`, which misclassified any
    unrecognized function (UPPER, ISNULL, custom UDFs, ...) as an aggregate.
    """
    if node is None:
        return False
    if node.find(*_AGG_NATIVE_TYPES) is not None:
        return True
    for anon in node.find_all(sqlglot.exp.Anonymous):
        name = (anon.this or "").lower() if isinstance(anon.this, str) else ""
        if name in _AGGREGATE_FUNCTION_NAMES:
            return True
    return False

def _has_ambiguous_columns(sql: str, tables: Optional[List[str]] = None,
                            live_cols: Optional[Dict[str, List[str]]] = None) -> bool:
    """
    Check if SQL has ambiguous column references, i.e. an unqualified
    column name that exists in more than one of the query's real tables
    (per `live_cols`, the actual schema for this request). Without
    `tables`/`live_cols` this degrades to "any unqualified column present"
    as a coarse fallback, since ambiguity can't be judged without schema.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)

        if not tables or not live_cols:
            return any(not col.table for col in expr.find_all(sqlglot.exp.Column))
        real_tables = {
            (t.name or "").upper()
            for t in expr.find_all(sqlglot.exp.Table)
            if t.name
        }
        col_to_owners: Dict[str, List[str]] = {}
        for tbl in tables:
            if tbl.upper() not in real_tables:
                continue
            for col in live_cols.get(tbl, []):
                col_to_owners.setdefault(col.lower(), []).append(tbl)

        for col in expr.find_all(sqlglot.exp.Column):
            if col.table:
                continue  # already qualified
            owners = col_to_owners.get(col.name.lower(), [])
            if len(owners) > 1:
                return True
        return False
    except Exception:
        return False

def _collect_add_chain(node) -> List:
    """Flatten a left-associative chain of sqlglot Add nodes into its leaf operands."""
    if isinstance(node, sqlglot.exp.Add):
        return _collect_add_chain(node.this) + _collect_add_chain(node.expression)
    return [node]

def _has_unsafe_plus_concat(sql: str) -> bool:
    """
    Detects the production bug pattern: Col1 + ' - ' + Col2 used to build a
    label/description string. SQL Server's `+` only concatenates when every
    operand is a string type; if any column is actually numeric (int,
    decimal, ...) — common for columns that merely *look* like codes, e.g.
    'Operation_Number' or 'Seq_Number' — the engine instead tries numeric
    addition and throws a conversion error at query time. A string literal
    anywhere in a `+` chain (e.g. ' - ') is a reliable signal the intent was
    concatenation, not arithmetic, regardless of the other operands' actual
    declared types (which aren't reliably knowable from live_cols alone).
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        for add in expr.find_all(sqlglot.exp.Add):
            if isinstance(add.parent, sqlglot.exp.Add):
                continue  
            operands = _collect_add_chain(add)
            if any(isinstance(o, sqlglot.exp.Literal) and o.is_string for o in operands):
                return True
        return False
    except Exception:
        return False

def _fix_unsafe_plus_concat(sql: str) -> str:
    """
    Companion to `_has_unsafe_plus_concat`. Rewrites every top-level `+`
    chain that contains a string literal into an equivalent CONCAT(...)
    call, which implicitly converts all arguments to string and never
    throws a type-conversion error — e.g.
      Col1 + ' - ' + Col2  ->  CONCAT(Col1, ' - ', Col2)
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        changed = False
        for add in list(expr.find_all(sqlglot.exp.Add)):
            if isinstance(add.parent, sqlglot.exp.Add):
                continue
            operands = _collect_add_chain(add)
            if not any(isinstance(o, sqlglot.exp.Literal) and o.is_string for o in operands):
                continue
            concat = sqlglot.exp.Anonymous(
                this="CONCAT",
                expressions=[o.copy() for o in operands],
            )
            add.replace(concat)
            changed = True
        if not changed:
            return sql
        result = expr.sql(dialect="tsql")
        log.info("[gen] Rewrote unsafe '+' string concatenation to CONCAT(): %s", result)
        return result
    except Exception as exc:
        log.warning("[gen] Failed to rewrite '+' concatenation (%s) — leaving SQL as-is", exc)
        return sql

def _has_group_by_error(sql: str) -> bool:
    """
    Check if SQL has GROUP BY coherence issues.
    Returns True if SELECT has non-aggregated columns not in GROUP BY.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return False
        group = expr.find(sqlglot.exp.Group)
        if not group:
            return False
        group_cols = set()
        for gb_col in group.find_all(sqlglot.exp.Column):
            group_cols.add((gb_col.table.lower() if gb_col.table else "", gb_col.name.lower()))
        for gb_id in group.find_all(sqlglot.exp.Identifier):
            group_cols.add(("", gb_id.name.lower()))
        
        has_agg = any(_contains_aggregate(sel_item) for sel_item in select.expressions)
        if not has_agg:
            return False
        
        for sel_expr in select.expressions:
            inner = sel_expr.this if isinstance(sel_expr, sqlglot.exp.Alias) else sel_expr
            if _contains_aggregate(inner):
                continue
            if isinstance(inner, (sqlglot.exp.Column, sqlglot.exp.Identifier)):
                tbl = (inner.table.lower() if hasattr(inner, "table") and inner.table else "")
                nm  = (inner.name.lower() if hasattr(inner, "name") else "")
                if (tbl, nm) not in group_cols and ("", nm) not in group_cols:
                    return True
        return False
    except Exception:
        return False

def _get_unknown_tables(sql: str, tables: List[str]) -> List[str]:
    """
    Return any table names referenced in `sql` that are not in the allowed
    `tables` list for this request. Catches model-hallucinated tables (e.g.
    inventing 'MFC_M_ITEM_TYPE_VSP' from a column like 'Item_Type_VSP_ID')
    before they're accepted as a successful generation. Unlike ambiguous
    columns / GROUP BY errors, this can't be auto-repaired locally — it
    needs to go back to the model with the real table list.
    """
    if not sql or not tables:
        return []
    allowed = {t.upper() for t in tables}
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        found = {
            (tnode.name or "").upper()
            for tnode in expr.find_all(sqlglot.exp.Table)
            if tnode.name
        }
        return sorted(found - allowed)
    except Exception as exc:
        log.warning("[validate] Failed to check table names: %s", exc)
        return []

def _get_unknown_columns(sql: str, tables: List[str], live_cols: Dict[str, List[str]]) -> List[str]:
    """
    Return any TableName.ColumnName references in `sql` where the column does
    not actually belong to that table according to `live_cols`. Catches cases
    like the model writing PRD_L_REWORK_ORDER.Serial_Number when Serial_Number
    actually belongs to PRD_L_REWORK_ORDER_SERIAL_NUMBER — SQL that parses fine
    but fails at execution time against SQL Server.
    """
    if not sql or not tables:
        return []
    allowed_tables = {t.upper() for t in tables}
    cols_by_table = {t.upper(): {c.upper() for c in live_cols.get(t, [])} for t in tables}
    unknown: List[str] = []
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        for col in expr.find_all(sqlglot.exp.Column):
            if not col.table:
                continue
            tbl = col.table.upper()
            nm = (col.name or "").upper()
            if tbl not in allowed_tables:
                continue  # unknown table already caught by _get_unknown_tables
            known_cols = cols_by_table.get(tbl)
            if known_cols and nm and nm not in known_cols:
                ref = f"{col.table}.{col.name}"
                if ref not in unknown:
                    unknown.append(ref)
        return unknown
    except Exception as exc:
        log.warning("[validate] Failed to check column names: %s", exc)
        return []

def _extract_first_valid_sql(text: str) -> Optional[str]:
    """Extract the first valid SQL SELECT statement from text."""
    if not text:
        return None
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
    return candidate if candidate else None

# ── Prompt Builder ──────────────────────────────────────────────────────────
def build_xiyansql_prompt(
    question: str,
    tables: List[str],
    join_clause: Optional[str],
    table_metadata: Dict,
    join_graph: Dict,
    lov_data: Dict,
    live_cols: Dict[str, List[str]],
    resolved_filters: Dict[str, str],
    lov_matches: Dict[str, Union[str, List[str]]],
    module_rules: str,
    intent: str,
    subgroup_name: str,
) -> str:
    """
    Build the prompt for XiYanSQL model.
    """
    prompt_parts = []
    
    # 1. System/role prompt
    prompt_parts.append(
        "You are an expert SQL generator for a manufacturing execution system (MES). "
        "Generate T-SQL queries for Microsoft SQL Server. "
        "Use the exact column names and table names shown below."
    )
    
    # 2. Module rules (domain guidance)
    if module_rules:
        prompt_parts.append(f"\nDomain rules:\n{module_rules}")
    
    # 3. Context
    prompt_parts.append(f"\nModule: {subgroup_name}")
    prompt_parts.append(f"Query intent: {intent}")
    
    # 4. Table schemas
    prompt_parts.append("\nAvailable tables and columns:")
    for table in tables:
        meta = table_metadata.get(table, {})
        prompt_parts.append(f"\nTable: {table}")
        prompt_parts.append(f"Business purpose: {meta.get('business_purpose', 'Not specified')}")
        prompt_parts.append(f"Row represents: {meta.get('row_represents', 'Not specified')}")
        
        # Columns
        cols = live_cols.get(table, [])
        if cols:
            prompt_parts.append("Columns (use only these exact names):")
            for col in cols:
                col_info = meta.get("columns", {}).get(col, {})
                if isinstance(col_info, dict):
                    desc = col_info.get("description", "")
                    dtype = col_info.get("data_type", "")
                    is_pk = " (PRIMARY KEY)" if col_info.get("is_pk") else ""
                    if desc:
                        prompt_parts.append(f"  - {col}: {dtype}{is_pk} -- {desc}")
                    else:
                        prompt_parts.append(f"  - {col}: {dtype}{is_pk}")
                else:
                    prompt_parts.append(f"  - {col}")
    
    # 5. Join relationships
    if join_graph:
        prompt_parts.append("\nKnown join relationships (use these for joins):")
        for from_table, targets in join_graph.items():
            for to_table, joins in targets.items():
                for join in joins:
                    if isinstance(join, dict):
                        from_col = join.get('from_column', '?')
                        to_col = join.get('to_column', '?')
                        prompt_parts.append(f"  {from_table}.{from_col} = {to_table}.{to_col}")
    
    # 5b. Resolved join clause (explicit, must-use path from upstream resolution)
    if join_clause:
        prompt_parts.append(
            "\nResolved JOIN path — use this relationship exactly when the "
            "question requires the related table:\n"
            f"{join_clause}"
        )

    # 6. LOV values
    if lov_data:
        prompt_parts.append("\nLOV (List of Values) - use numeric IDs, never string labels:")
        for table, lov in lov_data.items():
            for col, mapping in lov.items():
                examples = []
                for key, val in list(mapping.items())[:5]:  # Limit to 5 examples
                    if isinstance(val, dict):
                        label = val.get('description', str(key))
                        examples.append(f"'{label}' → {key}")
                    else:
                        examples.append(f"'{val}' → {key}")
                if examples:
                    prompt_parts.append(f"  {table}.{col}: {', '.join(examples)}")
                if len(mapping) > 5:
                    prompt_parts.append(f"  ... and {len(mapping) - 5} more values")
    
    # 7. Pre-resolved filters
    if resolved_filters:
        prompt_parts.append("\nPre-resolved filter values (use these exact values):")
        for col, val in resolved_filters.items():
            prompt_parts.append(f"  {col} = {val}")

    if lov_matches:
        prompt_parts.append("\nValues matched from the question text (use these exact values):")
        for col, val in lov_matches.items():
            if isinstance(val, list):
                if len(val) == 1:
                    prompt_parts.append(f"  {col} = {val[0]}")
                else:
                    vals_csv = ", ".join(str(v) for v in val)
                    prompt_parts.append(f"  {col} IN ({vals_csv})")
            else:
                prompt_parts.append(f"  {col} = {val}")

    # 8. Instructions - STRICT RULES
    prompt_parts.append(
        "\nSTRICT RULES - MUST FOLLOW:"
        "\n1. Use ONLY the exact column names listed above."
        "\n2. Every table reference must use the exact table name supplied in "
        "Available tables. Do not invent aliases unless absolutely required. "
        "Every column reference must be fully qualified using that exact table "
        "name, written as TableName.ColumnName (e.g., WO_Work_Order.Work_Order_ID)."
        "\n3. Do NOT use table aliases like 't', 'b', 'wo' in FROM or JOIN clauses. "
        "If you do add an alias (e.g. FROM PRD_L_REWORK_ORDER AS prd), every single "
        "reference to that table's columns anywhere in the query (SELECT, JOIN ON, "
        "WHERE, GROUP BY, ORDER BY) MUST use that alias too, never the real table "
        "name — SQL Server rejects the real name once an alias exists. WRONG: "
        "'FROM PRD_L_REWORK_ORDER AS prd ... SELECT PRD_L_REWORK_ORDER.Rework_Order_ID' "
        "(mixes alias in FROM with real name in SELECT — fails with 'multi-part "
        "identifier could not be bound'). Simplest and safest: don't alias at all, "
        "and use the real table name everywhere, per rule 2."
        "\n4. Always include a complete FROM clause with all required JOINs."
        "\n5. For numeric columns in WHERE conditions, use numbers without quotes."
        "\n6. For string columns, use UPPER() for case-insensitive comparisons."
        "\n7. Output only the SQL statement - no explanation, no markdown, no comments."
        "\n8. Do NOT use subqueries for LOV lookups - use the numeric IDs provided."
        "\n9. For a plain total/count question (e.g. 'total number of items', 'how many "
        "orders are there') that does NOT ask for a breakdown by category, return "
        "EXACTLY ONE ROW: SELECT COUNT(*) AS <alias> FROM <table> — with NO other "
        "columns in the SELECT list and NO GROUP BY. Adding the entity's own ID "
        "column and grouping by it turns a single total into one row per record "
        "(each showing count=1), which is WRONG for this kind of question. Only "
        "add GROUP BY when the user explicitly asks for a count 'per', 'by', or "
        "'for each' category/dimension (e.g. 'items per vendor', 'orders by status')."
        "\n10. NEVER use the + operator to join columns into a label/description "
        "string (e.g. Col1 + ' - ' + Col2). SQL Server's + tries numeric addition "
        "when either operand isn't varchar/nvarchar/char, and throws a conversion "
        "error — this happens even when the column LOOKS like a code (e.g. an "
        "'Operation_Number' or 'Seq_Number' column can be stored as int/decimal). "
        "ALWAYS use CONCAT(Col1, ' - ', Col2) instead when building any composite "
        "or human-readable string from two or more columns — CONCAT implicitly "
        "converts every argument to string and never throws a type error."
        "\n11. The SELECT list must contain ONLY the columns the question actually "
        "asks for (plus any column needed for an explicit ORDER BY/aggregation). "
        "Do NOT add extra descriptive columns 'for context' just because they "
        "exist in the table. E.g. for 'show items created this year', SELECT "
        "only the item name and the created-date column — not status, quantity, "
        "vendor, or other unrelated columns from the same table. A column used "
        "only internally to build a JOIN or GROUP BY does not need to appear in "
        "the SELECT list unless the question asked for it."
        "\n12. JOIN vs EXISTS:"
        "\n- If the user asks to DISPLAY, RETURN, LIST, SHOW, or SELECT columns "
        "from a related table, use JOIN and project those requested columns."
        "\n- Do NOT use EXISTS when columns from that related table are required "
        "in the SELECT list."
        "\n- Use EXISTS only when the related table is needed purely as a filter "
        "and no column from that table is requested in the result."
        "\n- If the question asks for parent records that HAVE matching child "
        "records, EXISTS is allowed."
        "\n- If the question asks for the child records themselves, JOIN is required."
        "\n- Follow the resolved JOIN path provided above whenever one exists."
        "\n13. When the user asks for information belonging to a related table "
        "(not just a filter on it), that related table MUST contribute columns "
        "to the SELECT list — do not join to it and then leave it unused in the "
        "output. Use the schema/columns listed above for each table to determine "
        "which column(s) actually represent the requested information; do not "
        "guess a column name or assume it lives on a different table."
        "\n14. RESULT TABLE / BUSINESS OBJECT RETRIEVAL:"
        "\n- When the question asks \"which\", \"what\", \"show\", \"list\", \"details\", "
        "or \"assigned to\" for a related business object, identify the table "
        "that represents that requested object as the RESULT TABLE."
        "\n- The RESULT TABLE must contribute one or more relevant columns to SELECT."
        "\n- Do not treat the RESULT TABLE as EXISTS-only filtering."
        "\n- Use the available column names and descriptions to select the columns "
        "that represent the requested business object."
        "\n- Example: \"Which routing is assigned to this item?\" → "
        "MFC_L_ITEM_ROUTING is the result table, not merely a filter table. → "
        "SELECT relevant routing columns from MFC_L_ITEM_ROUTING."
        "\n- Do not invent column names. Use only columns supplied in Available "
        "tables and columns metadata."
    )
    
    # 9. User question
    prompt_parts.append(f"\nUser question: {question}")
    
    # 10. Final instruction
    prompt_parts.append("\nGenerate the SQL query:")
    
    return "\n".join(prompt_parts)

# ── Ollama Client ──────────────────────────────────────────────────────────
async def call_ollama(prompt: str, max_tokens: int = _OLLAMA_MAX_TOKENS) -> Tuple[Optional[str], bool]:
    """
    Call Ollama with XiYanSQL model.
    Returns (response, was_truncated)
    """
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
            "stop": ["</s>", "<|im_end|>", "```", "Question:", "User:", "Explanation:"],
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OLLAMA_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=_OLLAMA_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    log.error("[ollama] Server returned HTTP %d", resp.status)
                    return None, False
                
                data = await resp.json()
                response = data.get("response", "").strip()
                done_reason = data.get("done_reason", "")
                was_truncated = (done_reason == "length")
                
                if not response:
                    log.warning("[ollama] Empty response from model")
                    return None, False

                if not response.upper().lstrip().startswith("SELECT"):
                   
                    extracted = _extract_first_valid_sql(response)
                    if extracted:
                        return extracted, was_truncated
                    log.warning("[ollama] Response doesn't contain valid SQL: %.80s", response)
                    return None, was_truncated
                
                return response, was_truncated
                
    except asyncio.TimeoutError:
        log.error("[ollama] Timeout after %.0fs", _OLLAMA_TIMEOUT)
        return None, False
    except Exception as exc:
        log.error("[ollama] Call failed: %s", exc)
        return None, False

# ── SQL Validation & Fixes ────────────────────────────────────────────────
def fix_ambiguous_columns(sql: str, tables: List[str], live_cols: Dict[str, List[str]]) -> str:
    """
    Attempt to fix ambiguous column references by qualifying them.
    """
    if not sql or not tables:
        return sql
    
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.WARN)
        col_to_tables: Dict[str, List[str]] = {}
        for tbl in tables:
            for col in live_cols.get(tbl, []):
                col_to_tables.setdefault(col.lower(), []).append(tbl)
        for col in expr.find_all(sqlglot.exp.Column):
            if col.table:
                continue  
            
            col_name = col.name.lower()
            if col_name in col_to_tables:
                owners = col_to_tables[col_name]
                if len(owners) == 1:
                    col.set("table", sqlglot.exp.Identifier(this=owners[0]))
                    log.info("[fix] Qualified ambiguous column '%s' with table '%s'", col.name, owners[0])
                elif len(owners) > 1:
                    log.warning(
                        "[fix] Cannot safely resolve ambiguous column '%s' — "
                        "multiple owners: %s",
                        col.name,
                        owners,
                    )
        
        return expr.sql(dialect="tsql")
    except Exception as exc:
        log.warning("[fix] Failed to fix ambiguous columns: %s", exc)
        return sql

def fix_group_by_coherence(sql: str) -> str:
    """
    Fix GROUP BY coherence issues.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql
        
        has_agg = any(_contains_aggregate(sel_item) for sel_item in select.expressions)
        if not has_agg:
            return sql
        
        group = expr.find(sqlglot.exp.Group)
        
        # Collect non-aggregated columns
        bare_cols = []
        for sel_expr in select.expressions:
            inner = sel_expr.this if isinstance(sel_expr, sqlglot.exp.Alias) else sel_expr
            if _contains_aggregate(inner):
                continue
            if isinstance(inner, (sqlglot.exp.Column, sqlglot.exp.Identifier)):
                bare_cols.append(inner.copy())
        
        if not bare_cols:
            return sql
        
        if group is None:
            # Add GROUP BY
            group_expr = sqlglot.exp.Group(expressions=bare_cols)
            expr.set("group", group_expr)
            log.info("[fix] Added GROUP BY with %d columns", len(bare_cols))
        else:
            # Add missing columns
            existing_keys = set()
            for gb_col in group.find_all(sqlglot.exp.Column):
                existing_keys.add((gb_col.table.lower() if gb_col.table else "", gb_col.name.lower()))
            for gb_id in group.find_all(sqlglot.exp.Identifier):
                existing_keys.add(("", gb_id.name.lower()))
            
            missing = []
            for col in bare_cols:
                tbl = (col.table.lower() if hasattr(col, "table") and col.table else "")
                nm  = (col.name.lower() if hasattr(col, "name") else "")
                if (tbl, nm) not in existing_keys and ("", nm) not in existing_keys:
                    missing.append(col)
            
            if missing:
                group.set("expressions", list(group.expressions) + missing)
                log.info("[fix] Added %d missing columns to GROUP BY", len(missing))
        
        return expr.sql(dialect="tsql")
    except Exception as exc:
        log.warning("[fix] Failed to fix GROUP BY: %s", exc)
        return sql

def _resolve_bare_group_col_table(expr, col_name: str, table_metadata: Dict) -> str:
    """
    Resolve which table in this query's own FROM/JOIN actually owns an
    unqualified GROUP BY column. Without this, a bare column (no table
    prefix) always misses in table_metadata.get(""), which silently made
    the PK-only/spurious-identifier guards below skip themselves on any
    GROUP BY column that wasn't fully qualified — even though the prompt's
    STRICT RULES require qualification, this is defense-in-depth for
    whatever the model emits anyway. Returns "" (unresolved) when the
    query has no FROM/JOIN tables, or the column name is owned by more
    than one of them — never guesses on ambiguity.
    """
    try:
        real_tables = {
            (t.name or "").upper()
            for t in expr.find_all(sqlglot.exp.Table)
            if t.name
        }
    except Exception:
        return ""
    owners = [
        t for t in real_tables
        if col_name.lower() in {
            c.lower() for c in (table_metadata.get(t, {}).get("columns", {}) or {}).keys()
        }
    ]
    return owners[0] if len(owners) == 1 else ""


def _has_pk_only_group_by(sql: str, table_metadata: Dict) -> bool:
    """
    Detects the exact bug pattern seen in production: the model emits an
    aggregate (COUNT/SUM/...) alongside a GROUP BY whose ONLY column(s) are
    the primary key of the table being grouped. Since a PK is unique per
    row, grouping by it always yields exactly one row per record — silently
    turning what should be a single scalar total ("total number of items")
    into N rows each showing e.g. count=1. A legitimate "group by category"
    query groups by a real business dimension (status, vendor, type) —
    never by the row's own identity column — so this check has no false
    positives against genuine per-category breakdowns.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        if not group:
            return False
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return False
        if not any(_contains_aggregate(sel_expr) for sel_expr in select.expressions):
            return False  # no aggregate — a normal GROUP BY, not this bug

        group_cols = list(group.find_all(sqlglot.exp.Column))
        if not group_cols:
            return False

        for gcol in group_cols:
            tbl = gcol.table or _resolve_bare_group_col_table(expr, gcol.name or "", table_metadata)
            col_name = gcol.name or ""
            meta = table_metadata.get(tbl) or table_metadata.get(tbl.upper()) or {}
            col_info = (meta.get("columns", {}) or {}).get(col_name, {})
            is_pk = isinstance(col_info, dict) and bool(col_info.get("is_pk"))
            if not is_pk:
                return False  
        return True  
    except Exception:
        return False


def _collapse_pk_group_by_to_scalar(sql: str) -> str:
    """
    Fixes the pattern flagged by `_has_pk_only_group_by`: strips the GROUP
    BY and drops every non-aggregated SELECT column, keeping only the
    aggregate expression(s) (and their aliases) — turning
      SELECT COUNT(T.Item_ID) AS Total, T.Item_ID FROM T GROUP BY T.Item_ID
    back into
      SELECT COUNT(T.Item_ID) AS Total FROM T
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return sql

        kept = [
            sel_expr for sel_expr in select.expressions
            if _contains_aggregate(sel_expr)
        ]
        if not kept:
            return sql  

        select.set("expressions", kept)
        expr.set("group", None)
        result = expr.sql(dialect="tsql")
        log.info("[gen] Collapsed spurious PK GROUP BY to scalar total: %s", result)
        return result
    except Exception as exc:
        log.warning("[gen] Failed to collapse PK GROUP BY (%s) — leaving SQL as-is", exc)
        return sql


def _has_spurious_identifier_in_group_by(sql: str, table_metadata: Dict) -> bool:
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
    chatbot_backend.py::_has_spurious_identifier_in_group_by.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        if not group:
            return False
        select = expr.find(sqlglot.exp.Select)
        if not select:
            return False
        if not any(_contains_aggregate(sel_expr) for sel_expr in select.expressions):
            return False

        group_cols = list(group.find_all(sqlglot.exp.Column))
        if not group_cols:
            return False

        pk_flags = []
        for gcol in group_cols:
            tbl = gcol.table or _resolve_bare_group_col_table(expr, gcol.name or "", table_metadata)
            col_name = gcol.name or ""
            meta = table_metadata.get(tbl) or table_metadata.get(tbl.upper()) or {}
            col_info = (meta.get("columns", {}) or {}).get(col_name, {})
            is_pk = isinstance(col_info, dict) and bool(col_info.get("is_pk"))
            pk_flags.append(is_pk)
        return any(pk_flags) and not all(pk_flags)
    except Exception:
        return False


def _strip_identifier_cols_from_group_by(sql: str, table_metadata: Dict) -> str:
    """
    Companion to _has_spurious_identifier_in_group_by. Removes only the
    spurious PK/identifier column(s) from GROUP BY — keeping real
    dimension columns intact — drops matching bare (non-aggregated)
    columns from SELECT so the result stays valid SQL, and, if a JOIN is
    present, switches COUNT(col) to COUNT(DISTINCT col) so the join's row
    fan-out can't inflate the total either.
    """
    try:
        expr = sqlglot.parse_one(sql, dialect="tsql", error_level=_sqlglot_errors.ErrorLevel.IGNORE)
        group = expr.find(sqlglot.exp.Group)
        select = expr.find(sqlglot.exp.Select)
        if not group or not select:
            return sql

        keep_group, drop_names = [], set()
        for gcol in group.find_all(sqlglot.exp.Column):
            was_bare = not gcol.table
            tbl = gcol.table or _resolve_bare_group_col_table(expr, gcol.name or "", table_metadata)
            col_name = gcol.name or ""
            meta = table_metadata.get(tbl) or table_metadata.get(tbl.upper()) or {}
            col_info = (meta.get("columns", {}) or {}).get(col_name, {})
            is_pk = isinstance(col_info, dict) and bool(col_info.get("is_pk"))
            if is_pk:
                drop_names.add((tbl.lower(), col_name.lower()))
                if was_bare:
                    drop_names.add(("", col_name.lower()))
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
                and not _contains_aggregate(sel_expr)
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
        log.info("[gen] Stripped spurious identifier column(s) from mixed GROUP BY: %s", result)
        return result
    except Exception as exc:
        log.warning("[gen] Failed to strip spurious identifier GROUP BY cols (%s) — leaving SQL as-is", exc)
        return sql


# ── Main Generation Flow ──────────────────────────────────────────────────
async def generate_sql_with_retries(
    question: str,
    tables: List[str],
    join_clause: Optional[str],
    intent: str,
    module_name: str,
    subgroup_name: str,
    live_cols: Dict[str, List[str]],
    resolved_filters: Dict[str, str],
    lov_matches: Dict[str, Union[str, List[str]]],
    reverse_hits: List[dict],
    table_metadata: Dict,
    join_graph: Dict,
    lov_data: Dict,
    module_prompt_rules: str,
    config: Dict,
) -> Dict:
    """
    Generate SQL with internal retries for common errors.
    """
    max_retries = config.get("max_retries", 3)
    retry_count = 0
    base_prompt = build_xiyansql_prompt(
        question=question,
        tables=tables,
        join_clause=join_clause,
        table_metadata=table_metadata,
        join_graph=join_graph,
        lov_data=lov_data,
        live_cols=live_cols,
        resolved_filters=resolved_filters,
        lov_matches=lov_matches,
        module_rules=module_prompt_rules,
        intent=intent,
        subgroup_name=subgroup_name,
    )
    
    last_sql = None
    last_error = None
    hint = ""  
    for attempt in range(max_retries):
        retry_count = attempt
        prompt = base_prompt + hint
        hint = "" 
        max_tokens = config.get("max_tokens", _OLLAMA_MAX_TOKENS)
        if attempt > 0:
            max_tokens = min(max_tokens + 200, 800)
        
        log.info("[gen] Attempt %d/%d - calling Ollama", attempt + 1, max_retries)
        
        response, was_truncated = await call_ollama(prompt, max_tokens)
        
        if not response:
            last_error = "No response from model"
            if attempt < max_retries - 1:
                hint = "\n\nERROR: Previous attempt produced no SQL. Please try again with a valid SQL query."
            continue
        
        # Extract SQL if needed
        if not response.upper().lstrip().startswith("SELECT"):
            extracted = _extract_first_valid_sql(response)
            if extracted:
                response = extracted
            else:
                last_error = "No valid SQL found in response"
                if attempt < max_retries - 1:
                    hint = "\n\nERROR: Previous response didn't contain a SELECT statement. Generate only the SQL."
                continue
        
        # ── Validate and fix SQL ──────────────────────────────────────────
        sql = response
        unknown_tables = _get_unknown_tables(sql, tables)
        if unknown_tables:
            log.warning("[gen] Unknown table(s) referenced: %s", unknown_tables)
            last_error = f"Unknown table(s): {', '.join(unknown_tables)}"
            if attempt < max_retries - 1:
                hint = (
                    f"\n\nERROR: Your previous SQL referenced table(s) that do not exist: "
                    f"{', '.join(unknown_tables)}. ONLY use these exact tables: "
                    f"{', '.join(tables)}. Do not invent lookup/type tables — if a value "
                    f"needs translation, use the ID column directly as instructed above."
                )
                continue
            log.error("[gen] Still referencing unknown table(s) after retries: %s", unknown_tables)
            return {
                "sql": "",
                "intent": intent,
                "retry_count": retry_count,
                "error": f"Model kept referencing non-existent table(s): {', '.join(unknown_tables)}",
            }

        unknown_columns = _get_unknown_columns(sql, tables, live_cols)
        if unknown_columns:
            log.warning("[gen] Unknown column(s) referenced: %s", unknown_columns)
            last_error = f"Unknown column(s): {', '.join(unknown_columns)}"
            if attempt < max_retries - 1:
                hint = (
                    f"\n\nERROR: Your previous SQL referenced columns that do not exist: "
                    f"{', '.join(unknown_columns)}. Use ONLY the exact columns provided "
                    f"for the available tables — do not assume a column lives on a table "
                    f"just because it seems related."
                )
                continue
            log.error("[gen] Still referencing unknown column(s) after retries: %s", unknown_columns)
            return {
                "sql": "",
                "intent": intent,
                "retry_count": retry_count,
                "error": f"Model kept referencing non-existent column(s): {', '.join(unknown_columns)}",
            }

        if _has_ambiguous_columns(sql, tables, live_cols):
            log.warning("[gen] Ambiguous columns detected - attempting fix")
            sql = fix_ambiguous_columns(sql, tables, live_cols)
            if _has_ambiguous_columns(sql, tables, live_cols):
                if attempt < max_retries - 1:
                    hint = "\n\nIMPORTANT: Your previous SQL had ambiguous column names. QUALIFY EVERY COLUMN with its table name: TableName.ColumnName"
                    continue
                log.warning("[gen] Still ambiguous after fix - proceeding anyway")

        if _has_unsafe_plus_concat(sql):
            log.warning("[gen] Unsafe '+' string concatenation detected — rewriting to CONCAT()")
            sql = _fix_unsafe_plus_concat(sql)

        # Check for GROUP BY errors
        if _has_group_by_error(sql):
            log.warning("[gen] GROUP BY coherence issue detected - attempting fix")
            sql = fix_group_by_coherence(sql)
            if _has_group_by_error(sql):
                if attempt < max_retries - 1:
                    hint = "\n\nIMPORTANT: Your previous SQL had GROUP BY issues. Add all non-aggregated SELECT columns to GROUP BY."
                    continue
                log.warning("[gen] Still has GROUP BY issues after fix - proceeding anyway")
        if _has_pk_only_group_by(sql, table_metadata):
            log.warning("[gen] GROUP BY groups only by table's own PK — collapsing to scalar total")
            sql = _collapse_pk_group_by_to_scalar(sql)

        elif _has_spurious_identifier_in_group_by(sql, table_metadata):
            log.warning("[gen] GROUP BY mixes a real dimension with a spurious joined-table identifier — stripping it")
            sql = _strip_identifier_cols_from_group_by(sql, table_metadata)

        # Validate SQL is SELECT only
        if not sql.upper().lstrip().startswith("SELECT"):
            last_error = "SQL doesn't start with SELECT"
            if attempt < max_retries - 1:
                hint = "\n\nERROR: Previous SQL didn't start with SELECT. Generate only SELECT statements."
            continue
        
        log.info("[gen] ✅ SQL generated successfully on attempt %d", attempt + 1)
        return {
            "sql": sql,
            "intent": intent,
            "retry_count": retry_count,
            "truncated": was_truncated,
        }
    
    log.error("[gen] All %d retries exhausted - last error: %s", max_retries, last_error)
    return {
        "sql": "",
        "intent": intent,
        "retry_count": retry_count,
        "error": f"Failed to generate valid SQL after {max_retries} attempts: {last_error}",
    }

# ── Request Queue ────────────────────────────────────────────────────────────
class _GenJob:
    __slots__ = ("kwargs", "future", "enqueued_at")

    def __init__(self, kwargs: dict, future: "asyncio.Future"):
        self.kwargs = kwargs
        self.future = future
        self.enqueued_at = time.monotonic()


_gpu_queue: "asyncio.Queue[_GenJob]" = asyncio.Queue(maxsize=_GPU_QUEUE_MAXSIZE)
_gpu_worker_tasks: List[asyncio.Task] = []
async def _gpu_worker(worker_id: int) -> None:
    """Pulls one job at a time, runs it to completion, resolves its future."""
    log.info("[queue] worker-%d started", worker_id)
    while True:
        job = await _gpu_queue.get()
        try:
            if job.future.cancelled():
                continue
            waited = time.monotonic() - job.enqueued_at
            log.info("[queue] worker-%d picked up job (waited %.1fs, qsize=%d)",
                      worker_id, waited, _gpu_queue.qsize())
            try:
                result = await generate_sql_with_retries(**job.kwargs)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - never let worker die
                log.exception("[queue] worker-%d job failed: %s", worker_id, exc)
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            _gpu_queue.task_done()


async def _submit_job(kwargs: dict) -> Dict:
    """
    Enqueue a generation job and wait for it to be processed in FIFO order.
    Raises HTTPException(503) if the queue is full or the wait times out.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future" = loop.create_future()
    job = _GenJob(kwargs, future)

    try:
        _gpu_queue.put_nowait(job)
    except asyncio.QueueFull:
        log.warning("[queue] rejected job — queue full (maxsize=%d)", _GPU_QUEUE_MAXSIZE)
        raise HTTPException(
            status_code=503,
            detail=f"GPU service busy — queue full ({_GPU_QUEUE_MAXSIZE} pending). Try again shortly.",
        )

    queue_position = _gpu_queue.qsize()
    log.info("[queue] job enqueued (position=%d)", queue_position)

    try:
        return await asyncio.wait_for(future, timeout=_GPU_QUEUE_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        log.error("[queue] job timed out after %.0fs waiting in queue", _GPU_QUEUE_WAIT_TIMEOUT)
        raise HTTPException(
            status_code=503,
            detail=f"GPU service busy — timed out after {_GPU_QUEUE_WAIT_TIMEOUT:.0f}s waiting for a slot.",
        )


# ── FastAPI App ─────────────────────────────────────────────────────────────
# Models
class GenerateSQLRequest(BaseModel):
    question: str
    tables: List[str]
    join_clause: Optional[str] = None
    intent: str
    module_name: str
    subgroup_name: str
    live_cols: Dict[str, List[str]]
    resolved_filters: Dict[str, str] = {}
    lov_matches: Dict[str, Union[str, List[str]]] = {}
    reverse_hits: List[dict] = []
    table_metadata: Dict = {}
    join_graph: Dict = {}
    lov_data: Dict = {}
    module_prompt_rules: str = ""
    config: Dict = {}

class GenerateSQLResponse(BaseModel):
    sql: str
    intent: Optional[str] = None
    retry_count: int = 0
    error: Optional[str] = None
    truncated: bool = False

# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    log.info("=" * 60)
    log.info("GPU SQL Generation Service Starting...")
    log.info(f"  Host: {_GPU_HOST}:{_GPU_PORT}")
    log.info(f"  Ollama URL: {_OLLAMA_URL}")
    log.info(f"  Model: {_OLLAMA_MODEL}")
    log.info(f"  Data Path: {_DATA_PATH}")
    log.info("=" * 60)
    
    # Check Ollama is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _OLLAMA_TAGS_URL,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    log.info(f"✅ Ollama available - Models: {models}")
                    
                    # Check if our model is available
                    if _OLLAMA_MODEL not in models:
                        log.warning(f"⚠️ Model '{_OLLAMA_MODEL}' not found in Ollama. Please pull it: ollama pull {_OLLAMA_MODEL}")
                else:
                    log.warning(f"⚠️ Ollama returned status {resp.status}")
    except Exception as e:
        log.warning(f"⚠️ Ollama not reachable: {e}")
    
    # Check data directory
    if not _DATA_PATH.exists():
        log.warning(f"⚠️ Data path not found: {_DATA_PATH}")
    else:
        log.info(f"✅ Data path exists: {_DATA_PATH}")

    # Start queue worker(s) — requests are processed strictly FIFO from here on
    for i in range(_GPU_QUEUE_WORKERS):
        _gpu_worker_tasks.append(asyncio.create_task(_gpu_worker(i)))
    log.info(f"✅ Started {_GPU_QUEUE_WORKERS} GPU queue worker(s), maxsize={_GPU_QUEUE_MAXSIZE}")

    yield

    log.info("GPU Service shutting down...")
    for task in _gpu_worker_tasks:
        task.cancel()
    for task in _gpu_worker_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

# ── Create App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="GPU SQL Generation Service",
    description="Generates SQL using XiYanSQL model on GPU",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check Ollama
    ollama_status = "unknown"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _OLLAMA_TAGS_URL,
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    ollama_status = "healthy"
                else:
                    ollama_status = f"error_{resp.status}"
    except Exception:
        ollama_status = "unreachable"
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model": _OLLAMA_MODEL,
        "ollama_status": ollama_status,
        "data_path": str(_DATA_PATH),
        "queue": {
            "pending": _gpu_queue.qsize(),
            "maxsize": _GPU_QUEUE_MAXSIZE,
            "workers": _GPU_QUEUE_WORKERS,
            "wait_timeout_secs": _GPU_QUEUE_WAIT_TIMEOUT,
        },
    }

@app.post("/generate-sql", response_model=GenerateSQLResponse)
async def generate_sql(request: GenerateSQLRequest):
    """Generate SQL from natural language question."""
    log.info("[request] Generating SQL for: %s", request.question[:80])
    log.info("[request] Tables: %s, Intent: %s", request.tables, request.intent)
    
    if not request.tables:
        raise HTTPException(status_code=400, detail="No tables provided")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    result = await _submit_job(dict(
        question=request.question,
        tables=request.tables,
        join_clause=request.join_clause,
        intent=request.intent,
        module_name=request.module_name,
        subgroup_name=request.subgroup_name,
        live_cols=request.live_cols,
        resolved_filters=request.resolved_filters,
        lov_matches=request.lov_matches,
        reverse_hits=request.reverse_hits,
        table_metadata=request.table_metadata,
        join_graph=request.join_graph,
        lov_data=request.lov_data,
        module_prompt_rules=request.module_prompt_rules,
        config=request.config,
    ))
    
    log.info("[response] SQL length: %d, retries: %d, error: %s",
             len(result.get("sql", "")), result.get("retry_count", 0),
             result.get("error", "none"))
    
    return result

@app.post("/generate-sql/stream")
async def generate_sql_stream(request: GenerateSQLRequest):
    """Stream the SQL generation process."""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        qsize = _gpu_queue.qsize()
        if qsize > 0:
            yield f"event: status\ndata: {json.dumps({'message': f'Queued — {qsize} request(s) ahead of you...'})}\n\n"
        else:
            yield f"event: status\ndata: {json.dumps({'message': 'Starting SQL generation...'})}\n\n"

        try:
            result = await _submit_job(dict(
                question=request.question,
                tables=request.tables,
                join_clause=request.join_clause,
                intent=request.intent,
                module_name=request.module_name,
                subgroup_name=request.subgroup_name,
                live_cols=request.live_cols,
                resolved_filters=request.resolved_filters,
                lov_matches=request.lov_matches,
                reverse_hits=request.reverse_hits,
                table_metadata=request.table_metadata,
                join_graph=request.join_graph,
                lov_data=request.lov_data,
                module_prompt_rules=request.module_prompt_rules,
                config=request.config,
            ))
        except HTTPException as exc:
            yield f"event: error\ndata: {json.dumps({'error': exc.detail})}\n\n"
            return

        if result.get("error"):
            yield f"event: error\ndata: {json.dumps(result)}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps(result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "gpu_service:app",
        host=_GPU_HOST,
        port=_GPU_PORT,
        workers=1,  
        log_level="info",
    )