from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

_DB_NAME: str = os.getenv("DB_NAME", "").strip()
if not _DB_NAME:
    raise RuntimeError("DB_NAME must be set in .env")


# ============================================================
# COLUMN RESOLUTION (LIGHTWEIGHT, NO HARDCODING)
# ============================================================

def _normalise(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _col_tokens(col_name: str) -> List[str]:
    parts = col_name.replace("_", " ")
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", parts)
    return [t.lower() for t in parts.split() if t]


def resolve_requested_column(question: str, tables: List[str], meta: dict) -> Optional[str]:
    q_tokens = set(_normalise(question))

    stop = {"id", "no", "num", "ref", "val", "the", "and", "for", "of", "a",
            "is", "in", "to", "by", "at", "on", "as", "be"}

    best, best_score = None, 0

    for table in tables:
        for col in meta.get(table, {}).get("columns", {}).keys():
            tokens = [t for t in _col_tokens(col) if t not in stop and len(t) >= 3]
            score = sum(1 for t in tokens if t in q_tokens)
            if score > best_score:
                best_score = score
                best = col

    return best if best_score >= 2 else None


# ============================================================
# ABSOLUTE RULES
# ============================================================

_ABSOLUTE_RULES = """\
RULES:
1. Output raw T-SQL only. No explanation. No markdown.
2. Start with SELECT (already written). End with ;
3. Use ONLY table and column names listed in SCHEMA. Never invent names.
4. Always qualify every column as TableName.ColumnName (for example, TableName.NameColumn). NEVER use short aliases like a., b., t. or any invented prefix.
5. Never SELECT * — always name specific columns.
6. Never include audit columns: cdate, cuser, udate, uuser, tenant_id, notes, remarks. This database uses LAMI_ prefixed audit columns: use LAMI_Status (not Status), LAMI_Cdate (not Cdate), LAMI_Udate (not Udate), LAMI_Cuser (not Cuser), LAMI_Uuser (not Uuser). Always use the LAMI_ prefixed version.
7. SQL Server / T-SQL syntax only: TOP N not LIMIT, GETDATE() not NOW(), ISNULL() not IFNULL().
8. Add WHERE filters ONLY when the user explicitly asks for them.
9. For lookup: select only the single target column, no COUNT.
10. Never assign AS aliases to tables in FROM or JOIN clauses. Write FROM TableName, not FROM TableName t or FROM TableName AS t.
11. Never generate SQL comments (-- or /* */).
12. Output executable SQL only.
13. Do not explain columns.
14. Do not copy schema descriptions into SQL.
15. For simple listing queries (no explicit ORDER BY requested), always add ORDER BY <primary_key_or_date_column> DESC to return newest records first.
16. NEVER write your own date/time WHERE filter — no CURDATE(), GETDATE(), YEAR(), STR_TO_DATE(), date string comparisons, or any date/time condition — even if the question mentions "today", "this month", "last week", or any other date/time reference. The system automatically injects the correct date filter after SQL generation. Simply write the query as if no date/time was mentioned in the question, and omit any date-related WHERE condition entirely. Do NOT invent a column to represent "today" or any other date reference.
"""


# ============================================================
# INTENT RULES — full SQL shape + one concrete example per intent
# ============================================================

_INTENT_RULES: Dict[str, str] = {

    "simple": """\
SQL shape: SELECT TOP 50 <4-6 columns> FROM <TableName> ORDER BY TableName.<pk_or_date_col> DESC;
Include: ID column, name/code column, key business columns.
Qualify every column as TableName.ColumnName. No table aliases.
Always end with ORDER BY <primary key or date column> DESC.
Example:
  SELECT TOP 50 TableName.IdColumn, TableName.NameColumn, TableName.StatusColumn
  FROM TableName
  ORDER BY TableName.IdColumn DESC;""",

    "conditional": """\
SQL shape: SELECT TOP 50 <relevant columns> FROM <TableName> WHERE <user condition> ORDER BY TableName.<pk_or_date_col> DESC;
Use only the condition the user explicitly stated. No extra filters.
Use UPPER(TableName.col) = UPPER('value') for text comparisons.
Qualify every column as TableName.ColumnName. No table aliases.
Always end with ORDER BY <primary key or date column> DESC.
Example:
  SELECT TOP 50 TableName.IdColumn, TableName.NameColumn, TableName.StatusColumn
  FROM TableName
  WHERE UPPER(TableName.StatusColumn) = UPPER('RELEASED')
  ORDER BY TableName.IdColumn DESC;""",

    "lookup": """\
SQL shape: SELECT TOP 1 <target column only> FROM <TableName> WHERE <key condition>;
Select ONLY the single target column. No extra columns. No COUNT.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT TOP 1 TableName.NameColumn
  FROM TableName
  WHERE TableName.IdColumn = 'VALUE-1234';""",

    "aggregation": """\
SQL shape: SELECT <aggregate function>(TableName.column) AS <label> FROM <TableName>;
Use COUNT(*) for row counts, SUM/AVG/MIN/MAX for numeric columns tagged [num].
Give the result a clear alias (e.g. AS TotalCount, AS AvgQty).
MANDATORY: You MUST use an aggregate function. A plain SELECT without SUM/COUNT is WRONG.
No table aliases. Qualify every column as TableName.ColumnName.
Example:
  SELECT COUNT(*) AS TotalWorkOrders
  FROM TableName;""",

    "group_by": """\
SQL shape: SELECT TableName.col, <aggregate> AS <label> FROM <TableName>
           GROUP BY TableName.col ORDER BY <aggregate> DESC;
Always include both the group column and the aggregate in SELECT.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT TableName.StatusColumn, COUNT(*) AS RowCount
  FROM TableName
  GROUP BY TableName.StatusColumn
  ORDER BY RowCount DESC;""",

    "top_bottom": """\
SQL shape: SELECT TOP <N> <key columns + metric column> FROM <TableName>
           ORDER BY TableName.<metric column> <DESC|ASC>;
Extract N from the user question. Default to 10 if not stated.
ORDER BY DESC for top/highest/most, ASC for bottom/lowest/least.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT TOP 10 TableName.IdColumn, TableName.NameColumn, TableName.QuantityColumn
  FROM TableName
  ORDER BY TableName.QuantityColumn DESC;""",

    "existence": """\
SQL shape: SELECT CASE WHEN EXISTS (
               SELECT 1 FROM <TableName> WHERE <condition>
           ) THEN 'Yes' ELSE 'No' END AS Result;
Use this exact shape. Do not use COUNT. Do not return rows.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT CASE WHEN EXISTS (
      SELECT 1 FROM TableName WHERE UPPER(TableName.StatusColumn) = UPPER('OPEN')
  ) THEN 'Yes' ELSE 'No' END AS Result;""",

    "time_based": """\
SQL shape: SELECT TOP 50 <relevant columns> FROM <TableName>
           WHERE <date column> >= <date expression> ORDER BY TableName.<date column> DESC;
T-SQL date functions: GETDATE(), DATEADD(day,-7,GETDATE()), CAST(GETDATE() AS DATE).
Map user intent:
  "today"        -> CAST(TableName.col AS DATE) = CAST(GETDATE() AS DATE)
  "this week"    -> TableName.col >= DATEADD(day, -7, GETDATE())
  "this month"   -> TableName.col >= DATEADD(month, -1, GETDATE())
  "last N days"  -> TableName.col >= DATEADD(day, -N, GETDATE())
Always ORDER BY the date column DESC.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT TOP 50 TableName.IdColumn, TableName.NameColumn, TableName.DateColumn
  FROM TableName
  WHERE TableName.DateColumn >= DATEADD(day, -7, GETDATE())
  ORDER BY TableName.DateColumn DESC;""",

    "sort_by": """\
SQL shape: SELECT TOP 50 <relevant columns including sort column>
           FROM <TableName> ORDER BY TableName.<sort column> <DESC|ASC>;
Include the column being sorted in the SELECT list.
Default sort direction: DESC unless user says ascending/oldest/lowest.
Qualify every column as TableName.ColumnName. No table aliases.
Example:
  SELECT TOP 50 TableName.IdColumn, TableName.NameColumn, TableName.QuantityColumn
  FROM TableName
  ORDER BY TableName.QuantityColumn DESC;""",

    "search": """\
SQL shape: SELECT TOP 50 <text column + key columns> FROM <TableName>
           WHERE TableName.<text column> LIKE '%<keyword>%'
           ORDER BY TableName.<pk_or_date_col> DESC;
Use UPPER(TableName.col) LIKE UPPER('%keyword%') for case-insensitive match.
Qualify every column as TableName.ColumnName. No table aliases.
Always end with ORDER BY <primary key or date column> DESC.
Example:
  SELECT TOP 50 TableName.IdColumn, TableName.NameColumn
  FROM TableName
  WHERE UPPER(TableName.NameColumn) LIKE UPPER('%PUMP%')
  ORDER BY TableName.IdColumn DESC;""",

"relational_lookup": """\
SQL shape: SELECT TOP 50 <business-relevant columns> FROM <JOIN clause exactly as given>
           ORDER BY <primary key of the primary entity> DESC;

GENERAL RULES:
1. Use the JOIN clause provided in the JOIN section exactly as written. Do NOT invent, modify, remove, or replace joins.
2. Qualify EVERY column as TableName.ColumnName. Never use short aliases. Never alias tables.
3. Select only the columns required to answer the user's question. Avoid unrelated business columns.
4. Treat the provided tables according to their semantic role:
   - Root/Master tables identify the primary business entity.
   - Detail/Child tables contain line-level, component-level, operation-level, attribute-level, or transaction-level information.
   - Bridge/Link tables are used only to establish relationships unless the user explicitly requests data from them.
   - Lookup tables should only be used when their descriptive values are required.
5. When both a root table and one or more related detail tables are provided, retrieve detailed information from the detail tables while using the root table only for identification, filtering, or context.
6. Never answer a detail-level request using only a master/root table when an appropriate detail table exists in the provided schema.
7. If the requested information exists in exactly one provided table, retrieve it from that table instead of selecting unrelated columns from another table.
8. Do not ignore any provided non-bridge table unless it is clearly irrelevant to the user's request.
9. Preserve the semantic relationships implied by the supplied JOIN clause. Do not simplify the query by removing detail tables.
10. Always end with ORDER BY <primary key of the primary entity> DESC unless the user explicitly requests another ordering.

Example:
SELECT TOP 50
    MainTable.IdColumn,
    MainTable.NameColumn,
    DetailTable.BusinessColumn,
    DetailTable.QuantityColumn
FROM MainTable
INNER JOIN DetailTable
ON MainTable.IdColumn = DetailTable.MainIdColumn
ORDER BY MainTable.IdColumn DESC;
""",

    "relational_aggregation": """\
DECISION — read the question carefully before choosing a shape:

IF the question asks for a LIST / HISTORY / DETAILS (words like: history, list, show, details,
records, entries, what are, which, trace, log, track, per serial, per SN) — use this shape:
  SELECT TOP 50 <key identifier col>, <descriptive cols from each table>
  FROM <JOIN clause exactly as given>
  WHERE <filter if stated>;
  Do NOT use SUM/COUNT/GROUP BY for listing queries.

IF the question asks for TOTALS / COUNTS / SUMMARY (words like: total, count, how many,
sum, average, per group, breakdown) — use this shape:
  SELECT TableName.group_col, AGG_FN(TableName.numeric_col) AS Label
  FROM <JOIN clause exactly as given>
  GROUP BY TableName.group_col
  ORDER BY Label DESC;
  MANDATORY: use SUM/COUNT, include GROUP BY, every non-aggregate col in GROUP BY.

RULES FOR BOTH SHAPES:
1. Use the JOIN clause from the JOIN section exactly as written. Do NOT invent joins.
2. Qualify EVERY column as TableName.ColumnName. No short aliases (no b., t., wo.).
7. Give every aggregate a clear AS alias (e.g. AS TotalQty, AS OrderCount).
8. Do NOT write a plain SELECT without SUM/COUNT/GROUP BY — that is WRONG for this intent.

Example:
  SELECT MainTable.GroupColumn,
         SUM(MainTable.QuantityColumn) AS TotalQuantity
  FROM MainTable
  INNER JOIN RelatedTable
    ON MainTable.RelatedIdColumn = RelatedTable.RelatedIdColumn
  GROUP BY MainTable.GroupColumn
  ORDER BY TotalQuantity DESC;""",

    "join": """\
SQL shape: SELECT TOP 50 <2-3 columns per table> FROM <JOIN clause exactly as given>;
Use the JOIN clause provided in the JOIN section exactly as written. Do not invent joins.
Qualify EVERY column as TableName.ColumnName. Never use short aliases. No AS aliases on tables.
Example:
  SELECT TOP 50 RelatedTable.NameColumn, MainTable.IdColumn, MainTable.NameColumn
  FROM RelatedTable
  INNER JOIN MainTable ON RelatedTable.RelatedIdColumn = MainTable.RelatedIdColumn;""",
}


# ============================================================
# AUDIT COLUMNS — always excluded from schema shown to LLM
# ============================================================

_AUDIT_SUFFIXES = ("cdate", "cuser", "udate", "uuser", "tenant_id", "notes", "remarks")


def _is_audit_col(col: str) -> bool:
    return any(col.lower().endswith(s) for s in _AUDIT_SUFFIXES)


def _schema_table_order(tables: List[str]) -> List[str]:
    """
    Return tables sorted so transaction / history tables come first.
    The LLM reads top-down, so the primary data table must appear before
    master/lookup tables — otherwise it anchors on master table columns.

    Priority (lowest number = shown first):
      0 — name contains _T_  (transaction tables, e.g. MFC_T_WORK_ORDER_SN_ROU_HISTORY)
      1 — name contains _L_  (link/log tables)
      2 — name contains _M_  (master tables)
      3 — everything else
    """
    def _rank(t: str) -> int:
        tu = t.upper()
        if "_T_" in tu:
            return 0
        if "_L_" in tu:
            return 1
        if "_M_" in tu:
            return 2
        return 3

    return sorted(tables, key=_rank)




def _build_join_block(join_clause: str) -> str:
    return (
        f"\nJOIN:\n"
        f"FROM {join_clause}\n"
        f"Use this FROM clause exactly as written. Do not add AS aliases to any table.\n"
        f"Qualify every column as TableName.ColumnName using the full table names above.\n"
    )


# ============================================================
# TARGET COLUMN BLOCK
# ============================================================

def _build_target_col_block(col: str) -> str:
    return f"\nTARGET COLUMN: {col}\nSelect only this.\n"


# ============================================================
# SCHEMA SECTION
# ============================================================

def _col_type_hint(col_meta) -> str:
    """Return a short type tag for a column — helps LLM write correct comparisons."""
    if not isinstance(col_meta, dict):
        return ""
    dtype = (col_meta.get("data_type") or "").lower()
    if any(t in dtype for t in ("int", "decimal", "float", "numeric", "money")):
        return "[num]"
    if any(t in dtype for t in ("date", "time", "datetime")):
        return "[date]"
    if any(t in dtype for t in ("varchar", "nvarchar", "char", "nchar", "text")):
        return "[str]"
    return ""


def _build_schema_section(
    tables: List[str],
    meta: dict,
    lov_context_fn,
    narrow_index: Optional[Dict[str, dict]] = None,
    live_columns: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Build the SCHEMA block shown to the LLM.

    Column resolution priority (highest → lowest):
      1. live_columns  — real column names fetched live from the DB via
                         ``SELECT TOP 0 * FROM <table>``.  Always correct.
      2. meta          — columns from mfc.json / schema_raw.json.  Used when
                         live_columns is absent or the fetch failed for a table.

    When narrow_index (built from mfc_narrow.json) is provided each table
    section is further enriched with:
      - plain-English column descriptions  (LLM knows what each col means)
      - typical query patterns             (LLM knows which cols normally needed)
      - model routing hint                 (direct guidance for this table)

    Without narrow_index the function falls back to the original bare
    column-name list — fully backwards-compatible.
    """
    lines = []
    live_columns = live_columns or {}
    ordered_tables = _schema_table_order(tables)
    for t in ordered_tables:

        if t in live_columns and live_columns[t]:
            raw_cols = live_columns[t]
            business_cols = [c for c in raw_cols if not _is_audit_col(c)]
            meta_cols = meta.get(t, {}).get("columns", {})
        elif t in meta:
            raw_cols = list(meta[t]["columns"].keys())
            business_cols = [c for c in raw_cols if not _is_audit_col(c)]
            meta_cols = meta[t]["columns"]
        else:
            continue

        narrow          = (narrow_index or {}).get(t, {})
        col_descs       = narrow.get("columns", {})
        query_patterns  = narrow.get("query_patterns", [])
        routing_hint    = narrow.get("model_routing_hint", "")

        if col_descs:
            lines.append(f"{t}:")
            for col in business_cols:
                col_meta = meta_cols.get(col, {})
                type_hint = _col_type_hint(col_meta)
                desc = col_descs.get(col, "")
                lines.append(f"  {col}{type_hint}")
        else:
            col_list = []
            for col in business_cols:
                col_meta = meta_cols.get(col, {})
                type_hint = _col_type_hint(col_meta)
                col_list.append(f"{col}{type_hint}")
            lines.append(f"{t}: {', '.join(col_list)}")

        if query_patterns:
            lines.append("  Query patterns:")
            for qp in query_patterns:
                lines.append(f"    - {qp}")

        if routing_hint:
            lines.append(f"  Guidance: {routing_hint}")

        lov = lov_context_fn(t)
        if lov:
            lines.append(lov)

    return "\n".join(lines)


# ============================================================
# MAIN BUILDER
# ============================================================

def build_prompt(
    tables: List[str],
    question: str,
    meta: dict,
    lov_context_fn,
    join_clause: Optional[str] = None,
    intent: str = "simple",
    target_column: Optional[str] = None,
    extra_hint: str = "",
    narrow_index: Optional[Dict[str, dict]] = None,
    live_columns: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Build the full LLM prompt for NL2SQL.

    Parameters
    ----------
    tables         : tables selected for this query
    question       : raw user question
    meta           : schema metadata dict (from mfc.json)
    lov_context_fn : callable(table_name) -> LOV context string
    join_clause    : pre-built JOIN clause (relational queries)
    intent         : detected query intent
    target_column  : specific column to look up (lookup intent only)
    extra_hint     : freeform text appended at end of system block.
    narrow_index   : flat { table_name -> mfc_narrow entry } dict.
    live_columns   : {table_name: [col1, col2, ...]} from live schema fetcher.
    """

    if intent == "lookup" and not target_column:
        target_column = resolve_requested_column(question, tables, meta)

    schema       = _build_schema_section(
        tables, meta, lov_context_fn,
        narrow_index=narrow_index,
        live_columns=live_columns,
    )
    intent_block = _INTENT_RULES.get(intent, _INTENT_RULES["simple"])

    join_block   = _build_join_block(join_clause) if join_clause and len(tables) > 1 else ""
    target_block = _build_target_col_block(target_column) if target_column else ""
    hint_block   = f"\nIMPORTANT:\n{extra_hint.strip()}\n" if extra_hint and extra_hint.strip() else ""

    system = (
        f"You are a T-SQL expert for database: {_DB_NAME}.\n"
        f"{_ABSOLUTE_RULES}\n\n"
        f"QUERY TYPE: {intent}\n"
        f"INSTRUCTIONS:\n{intent_block}\n\n"
        f"{hint_block}"
        f"ALLOWED TABLES: {', '.join(tables)}\n\n"
        f"SCHEMA (use ONLY these columns):\n{schema}"
        f"{join_block}"
        f"{target_block}"
    )

    _LISTING_SIGNALS = re.compile(
        r'\b(history|histories|list|listing|show|detail|details|record|records|'
        r'entry|entries|trace|log|track|per serial|per sn|what are|which)\b',
        re.IGNORECASE,
    )
    _AGG_INTENTS   = {"aggregation", "relational_aggregation"}
    _GROUP_INTENTS = {"group_by"}

    _is_listing_question = bool(_LISTING_SIGNALS.search(question))

    _TOP50_INTENTS = {"relational_lookup", "join", "relational_aggregation"}

    if intent in _AGG_INTENTS and not _is_listing_question:
        assistant_prefix = "SELECT SUM("
    elif intent in _GROUP_INTENTS:
        assistant_prefix = "SELECT"
    elif intent in _TOP50_INTENTS:
        assistant_prefix = "SELECT TOP 50 "
    else:
        assistant_prefix = "SELECT"

    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )