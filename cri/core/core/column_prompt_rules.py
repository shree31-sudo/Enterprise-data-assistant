

from __future__ import annotations

from typing import Dict, List, Optional


COLUMN_SELECTION_SYSTEM_RULES = """You are a COLUMN SELECTOR for a SQL generation pipeline. You do NOT write SQL.

TASK: Given a user question and a catalogue of tables with their real column names
(and short descriptions), decide which columns from EACH table are needed to answer
the question.

HARD RULES:
1. Choose columns ONLY from the exact names given in the catalogue below. Never invent,
   abbreviate, guess, or pluralize a column name that is not listed verbatim.
2. Every table listed in the catalogue MUST appear as a key in your output, even if you
   only select one column for it.
3. Prefer precision over recall: include a column only if the question needs it to
   filter, group, sort, or display the answer. Do not dump the whole table.
4. If the question is ambiguous about which columns to show, include the columns that
   most directly name/identify the row (name/code/title-type columns) for that table.
5. Do NOT include primary keys, foreign keys, audit columns (created_by, updated_by,
   tenant, etc.) unless the question explicitly asks about them — a separate
   downstream step adds mandatory keys automatically, so you do not need to worry
   about join integrity.
6. Output STRICT JSON ONLY — a single flat object mapping table name to a list of
   column name strings. No markdown fences, no prose, no comments, no trailing text.

OUTPUT FORMAT (exactly this shape, nothing else):
{"TABLE_A": ["col1", "col2"], "TABLE_B": ["col3"]}
"""

_FEW_SHOT_EXAMPLE = """EXAMPLE

Question: "Show me the item name and status for all active items"

Catalogue:
### MFC_M_ITEM
- Item_Code: unique code for the item
- Item_Name: display name of the item
- Item_Description: long text description
- Created_Date: date the row was created

### MFC_L_ITEM_ATTRIBUTE
- Item_Status: active/inactive flag for the item
- Attribute_Value: generic attribute value

Output:
{"MFC_M_ITEM": ["Item_Name"], "MFC_L_ITEM_ATTRIBUTE": ["Item_Status"]}
"""


def _format_table_catalogue(
    tables: List[str],
    table_metadata: Dict[str, Dict],
    desc_max_words: Optional[int] = None,
) -> str:
    """
    Render a compact per-table column catalogue block:

        ### TABLE_NAME
        - col_a: description
        - col_b: description

    desc_max_words: if set, each description is truncated to this many
    words (keeps the prompt short for a small model — mirrors
    COLUMN_SELECTION_DESC_MAX_WORDS in .env).
    """
    blocks: List[str] = []
    for tname in tables:
        meta = table_metadata.get(tname, {}) or {}
        columns: Dict[str, str] = meta.get("columns", {}) or {}

        if not columns:
          
            bare = meta.get("column_names", []) or []
            columns = {c: "" for c in bare}

        lines = [f"### {tname}"]
        if not columns:
            lines.append("- (no column metadata available)")
        else:
            for col, desc in columns.items():
                desc = (desc or "").strip()
                if desc and desc_max_words:
                    words = desc.split()
                    if len(words) > desc_max_words:
                        desc = " ".join(words[:desc_max_words]) + "..."
                if desc:
                    lines.append(f"- {col}: {desc}")
                else:
                    lines.append(f"- {col}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_column_selection_prompt(
    question: str,
    tables: List[str],
    table_metadata: Dict[str, Dict],
    intent: Optional[str] = None,
    include_few_shot: bool = True,
    desc_max_words: Optional[int] = None,
) -> str:
    """
    Compose the full prompt sent to the column-selection model.

    table_metadata: {table_name: {"columns": {col: desc, ...}, ...}}
                     (this is the per-table dict shape produced by
                     mfc_narrow.json / NARROW_TABLE_INDEX)
    desc_max_words: caps each column description's length (see
                     _format_table_catalogue).
    """
    catalogue = _format_table_catalogue(tables, table_metadata, desc_max_words=desc_max_words)

    parts: List[str] = [COLUMN_SELECTION_SYSTEM_RULES]
    if include_few_shot:
        parts.append(_FEW_SHOT_EXAMPLE)

    intent_line = f"\nIntent hint: {intent}\n" if intent else ""

    parts.append(
        "NOW DO THE REAL TASK.\n\n"
        f"Question: \"{question.strip()}\"\n"
        f"{intent_line}\n"
        f"Catalogue:\n{catalogue}\n\n"
        "Output:"
    )
    return "\n\n".join(parts)