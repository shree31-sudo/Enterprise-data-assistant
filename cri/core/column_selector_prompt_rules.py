
from __future__ import annotations

from typing import Dict

# ══════════════════════════════════════════════════════════════════════════
# Rules that apply to every module, every table, every question.
# ══════════════════════════════════════════════════════════════════════════
GENERAL_RULES: str = """
- Use table and column names EXACTLY as given in the catalog. Never invent a column.
- pipeline_columns = every column needed to build correct SQL: filters, joins, sort, group-by,
  and anything the question is asking to see. Prefer including a borderline column over dropping it —
  a missing pipeline column breaks SQL generation, an extra one does not.
- response_columns = the subset of pipeline_columns worth showing the end user in the final answer.
  Pure join/foreign-key columns used only to stitch tables together do not belong here unless the
  question is specifically about that relationship. Audit columns (created/updated by, created/updated
  date, tenant, internal flags) do not belong here unless explicitly asked about.
- Every identifier (primary key) column stays in pipeline_columns even when the question doesn't name it —
  it is frequently needed to disambiguate rows or drive a join.
- Prefer a human-readable name/code/description column over a raw numeric ID whenever both exist and the
  table role is being displayed to the user, not just referenced for a join.
- Status/state columns belong in pipeline_columns whenever the question implies filtering by
  active/valid/current/inactive/expired, even when the word "status" itself never appears.
- Date/time columns belong in pipeline_columns whenever the question implies a time window, ordering by
  recency, or "latest"/"current"/"as of" phrasing, even when no explicit date column is named.
- When a table has very few columns (a lookup/master table), default to including most non-audit
  columns in pipeline_columns rather than trying to narrow — the cost of narrowing wrongly on a small
  table is high relative to the savings.
- When a table is wide (many columns), be deliberate: only include what the join/filter/display actually
  needs, not everything that exists.
""".strip()


# ══════════════════════════════════════════════════════════════════════════
# Per-module structural guidance — about the SCHEMA and its relationships,
# ══════════════════════════════════════════════════════════════════════════
MODULE_RULES: Dict[str, str] = {
    "items": (
        "Item tables are structured around one master (item identity) with several linked attribute/"
        "property tables. The item's name/description column is a response_column essentially always — "
        "an item answer that only shows an ID is incomplete. Plant linkage is a pipeline_column whenever "
        "more than one table is involved, since item data is frequently plant-scoped, but it only becomes "
        "a response_column when plant is part of what's being displayed, not just filtered on."
    ),
    "bom": (
        "BOM tables are inherently parent/child (header -> line items, parent item -> child/component "
        "item). Both the parent and child item identifying columns are always pipeline_columns. The "
        "child item's descriptive column is preferred over its raw ID in response_columns; the parent "
        "is response-worthy only when multiple parents could be in the result set. Quantity/usage "
        "columns on line-item tables are pipeline_columns by default since BOM data is rarely useful "
        "without them, and become response_columns whenever more than a single row is returned."
    ),
    "routing": (
        "Routing tables are ordered by an operation-sequence column — that column is always a "
        "pipeline_column so ORDER BY / sequencing works, and becomes a response_column whenever more "
        "than one operation row is returned. Operation description columns are preferred over raw "
        "operation codes in response_columns; codes stay pipeline-only unless multiple operations share "
        "a description and disambiguation is needed."
    ),
    "erp_operation": (
        "Small master/lookup table — default to including most non-audit columns in pipeline_columns "
        "rather than narrowing. Operation code and operation description are a pair: if one is a "
        "response_column, the other should be too."
    ),
    "la_operation": (
        "Small master/lookup table, same pattern as erp_operation — default to including most non-audit "
        "columns in pipeline_columns. The operation type/category column stays in pipeline_columns even "
        "when not directly referenced, since it is frequently the disambiguator between similarly-named "
        "operations."
    ),
    "uom": (
        "Pure lookup table. pipeline_columns should include both the UOM code and UOM description "
        "columns. response_columns should prefer the description over the raw code."
    ),
    "mbom_operations": (
        "The widest tables in the schema — be deliberate rather than inclusive here. Traceability "
        "identifier columns (serial/packet/work-order-linked) are pipeline_columns whenever the tables "
        "in play are IME/execution tables, even if the question refers to the entity indirectly (e.g. "
        "'that unit', 'this batch') rather than by column name. Operation-level configuration columns "
        "are pipeline_columns only when configuration/setup is the subject of the question, not for "
        "simple lookup questions — including them by default on this module's wide tables does more "
        "harm than good."
    ),
}

DEFAULT_MODULE_RULE: str = (
    "No module-specific guidance is defined for this module yet — fall back to GENERAL_RULES only. "
    "Prefer narrower response_columns over wider ones when uncertain: a follow-up question is cheap, "
    "a cluttered answer is not."
)


def get_module_rule(module_name: str) -> str:
    """Structural guidance for one module. Always non-empty."""
    return MODULE_RULES.get((module_name or "").strip(), DEFAULT_MODULE_RULE)


def build_rule_text(module_name: str) -> str:
    """GENERAL_RULES + this module's rule, concatenated for prompt injection."""
    return f"{GENERAL_RULES}\n\nModule-specific ({module_name or 'unknown'}):\n{get_module_rule(module_name)}"
