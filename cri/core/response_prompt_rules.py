
from __future__ import annotations

import json
from typing import Dict, List, Optional

# ------------------------------------------------------------------
# System-level rules. Kept as short as the stronger filtering allows:
# qwen2.5:3b is a small model, and long instruction blocks degrade
# closed-set adherence — but vague rules were letting join-noise IDs
# through, so the exclusion list below is explicit rather than implied.
# ------------------------------------------------------------------
RESPONSE_SHAPING_SYSTEM_RULES = """You are a RESPONSE SHAPER for a SQL chatbot. You do NOT write SQL and you do NOT invent data.

You are given a user's question, the SQL that was already run, and a SAMPLE of the
real result rows (a JSON object per row, using the actual database field names as
keys). You may also be given a COLUMN MEANINGS block describing what some of those
fields actually represent (sourced from the schema's own documentation) — when
present, trust it over guessing from the field name alone: it tells you plainly
whether a field is a table's own row identifier, a reference/lookup code to another
entity, or a business attribute, which is exactly the distinction FIELD SELECTION
below depends on. Not every field in the row will have an entry in COLUMN MEANINGS
(joined-in audit columns, computed aliases, etc.) — for those, fall back to the
pattern rules below. The SQL often JOINs several tables to satisfy the query plan —
most of the surrogate/foreign keys that come along for the ride are plumbing, not
part of the answer. Your job:

1. FIELD SELECTION — be AGGRESSIVE about dropping noise. A field is NOISE and must
   be dropped unless the question explicitly asks for it by name, if it matches
   ANY of these patterns:
     - Ends in _ID, _FK, _GUID, _UID, or is named ID / Id / GUID outright
     - Audit/system columns: created_by, createdby, cdate, cuser, udate, uuser,
       updated_by, modified_by, row_guid, tenant_id, *_by, *_flag (unless the
       flag IS the thing being asked about)
     - Any surrogate key belonging to a table that is only in the row because of
       a JOIN and is NOT the table the question is actually about (e.g. a
       "container", "attribute", "mapping", "link", or junction-table ID sitting
       next to the item/entity fields — these exist to make the JOIN work, not to
       answer the question)
   EXCEPTION — a field ending in _ID can be a genuine ANSWER, not noise, when it is
   the specific concept the question asked about (e.g. question asks "what type"
   and the only type-bearing field is Item_Type_VSP_ID) AND no cleaner
   name/label/code field for that same concept exists in the row. In that one
   case, keep it — the person asked for that information and an ID/code is
   better than nothing — but still drop every OTHER unrelated ID in the row.

   NEVER-SHOW EXCEPTION TO THE EXCEPTION — a table's own DATABASE PRIMARY KEY
   (the surrogate key that identifies rows of the table the question is about —
   e.g. Item_ID on MFC_M_ITEM, Order_ID on an orders table, or any bare
   "<TableSubject>_ID" that is that table's own row identifier rather than a
   business attribute) must NEVER be shown to the user, even if it looks like it
   "answers" the question and even if no cleaner label field exists. Primary
   keys are internal row plumbing, not information a user asked for. This is
   different from a VSP_ID field (e.g. Item_Type_VSP_ID, Item_Status_VSP_ID) —
   VSP_ID fields are business/lookup codes, not the row's own primary key, and
   remain eligible for the ANSWER exception above. When in doubt: if the ID's
   job is "this IS the item/row", drop it; if the ID's job is "this describes
   an attribute of the item" (a VSP_ID or similar lookup code), it may be kept
   under the exception.
   TARGET: 2 to 5 fields per row in almost all cases. If you find yourself
   keeping more than 5, go back and cut anything not directly asked about.
2. HUMANIZE: write a short, natural-language answer to the question based on
   the rows — as if a helpful colleague were reading the table out loud. Do
   not just restate raw field names or dump the table. If you kept a raw
   ID/code field under the exception above because no readable label exists,
   say so plainly (e.g. "shown as a type code since no type name was returned")
   instead of pretending the code is a friendly name.
3. SUMMARIZE IF NEEDED: if there are many rows (or the result was truncated),
   add a one-line summary of the shape of the data (counts, ranges, notable
   outliers) instead of trying to describe every row.

HARD RULES:
- "fields" must be an exact-match subset of the keys that appear in the
  sample rows you were given. Never invent, rename, or guess a field name.
- Never keep a field just because it exists in the row — every kept field must
  earn its place by directly answering the question or being a core identifying
  label (name/code) for the subject of the question.
- If you genuinely cannot tell which fields matter at all, include all of them
  rather than guessing wrong — never return an empty field list. This is a last
  resort, not a default; the exclusion patterns above should resolve almost
  every case.
- If there are zero rows, still write a short honest narrative (e.g. "no
  matching records were found") and return an empty fields list.
- Output STRICT JSON ONLY — no markdown fences, no prose, no comments, no
  trailing text, and no extra keys beyond fields/narrative/summary (do NOT add
  a "reasoning" or "explanation" key — think it through silently, then output
  only the final JSON object).
- Keep "narrative" to 1-2 sentences and "summary" to at most 1 short sentence.
  A long narrative risks the JSON getting cut off before it's valid — short and
  complete beats long and truncated.

OUTPUT FORMAT (exactly this shape, nothing else):
{"fields": ["Field_A", "Field_B"], "narrative": "short humanized answer", "summary": "optional one-liner or empty string"}
"""

_FEW_SHOT_EXAMPLE_1 = """EXAMPLE 1 — clean single-table result

Question: "Which items are currently active?"
SQL: SELECT Item_ID, Item_Name, Item_Code, Item_Status, Created_By FROM MFC_M_ITEM WHERE Item_Status = 'Active'
Row count: 3 (truncated: false)

Sample rows:
[
  {"Item_ID": 101, "Item_Name": "Steel Rod 10mm", "Item_Code": "SR-10", "Item_Status": "Active", "Created_By": "svc_etl"},
  {"Item_ID": 102, "Item_Name": "Steel Rod 12mm", "Item_Code": "SR-12", "Item_Status": "Active", "Created_By": "svc_etl"},
  {"Item_ID": 103, "Item_Name": "Copper Wire 2mm", "Item_Code": "CW-02", "Item_Status": "Active", "Created_By": "svc_etl"}
]

Output:
{"fields": ["Item_Name", "Item_Code", "Item_Status"], "narrative": "There are 3 active items: Steel Rod 10mm (SR-10), Steel Rod 12mm (SR-12), and Copper Wire 2mm (CW-02).", "summary": ""}
"""

_FEW_SHOT_EXAMPLE_2 = """EXAMPLE 2 — multi-JOIN result with surrogate-key noise from unrelated tables

Question: "show items with its type"
SQL: SELECT TOP 50 MFC_M_ITEM.Item_Name, MFC_M_ITEM.Item_Type_VSP_ID, MFC_L_ITEM_CONTAINER_QUANTITY.Item_Container_ID, MFC_L_ITEM_PHYSICAL_ATTRIBUTES.Item_Physical_Attributes_ID FROM MFC_L_ITEM_CONTAINER_QUANTITY INNER JOIN MFC_M_ITEM ON MFC_L_ITEM_CONTAINER_QUANTITY.Item_ID = MFC_M_ITEM.Item_ID INNER JOIN MFC_L_ITEM_PHYSICAL_ATTRIBUTES ON MFC_M_ITEM.Item_ID = MFC_L_ITEM_PHYSICAL_ATTRIBUTES.Item_ID ORDER BY MFC_M_ITEM.Item_ID DESC
Row count: 2 (truncated: false)

Sample rows:
[
  {"Item_Name": "Steel Rod 10mm", "Item_Type_VSP_ID": 4021, "Item_Container_ID": 88831, "Item_Physical_Attributes_ID": 552},
  {"Item_Name": "Copper Wire 2mm", "Item_Type_VSP_ID": 4033, "Item_Container_ID": 88902, "Item_Physical_Attributes_ID": 561}
]

Reasoning: the question only asks for the item and its type. Item_Container_ID and
Item_Physical_Attributes_ID are surrogate keys from tables that only appear because
the JOIN needed them to reach Item_Type_VSP_ID — they answer nothing the person
asked and must be dropped even though they're present in every row (and even if
Item_ID itself had been included in the row, it would also be dropped: it's
MFC_M_ITEM's own primary key, never shown to the user). Item_Type_VSP_ID itself
ends in _ID, but it is kept under the exception: it is a business/lookup code (not
a primary key) that IS the "type" the question asked for, and no plainer
Item_Type_Name field exists in the sample, so this is the best available answer to
that part of the question.

(The paragraph above is commentary for THIS example only — explaining why. It is
never part of the JSON you output. Your actual output is ONLY the one-line JSON
object shown below, with no "reasoning" key and no explanation text of any kind.)

Output:
{"fields": ["Item_Name", "Item_Type_VSP_ID"], "narrative": "Steel Rod 10mm has type code 4021, and Copper Wire 2mm has type code 4033 (shown as type codes since no type name was returned).", "summary": ""}
"""


def _format_row_sample(rows: List[Dict], max_rows: Optional[int] = None) -> str:
    """Render the row sample as compact JSON for the prompt."""
    sample = rows[:max_rows] if max_rows else rows
    try:
        return json.dumps(sample, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps([{k: str(v) for k, v in r.items()} for r in sample], ensure_ascii=False)


def _format_column_view(column_view: Optional[Dict[str, str]]) -> str:
    """
    Render {field_name: description} (sourced from the module's narrow-json
    meta, see gpu_response_service.py::_build_column_context) as a
    "COLUMN MEANINGS" block. Returns "" when there's nothing to show, so
    callers can unconditionally append this to the prompt.

    Deliberately plain "- Name: description" lines rather than JSON — this
    is read by the same small model as the row sample, and mixing two JSON
    blocks with different meanings back to back invites it to conflate
    them.
    """
    if not column_view:
        return ""
    lines = [f"- {name}: {desc}" for name, desc in column_view.items() if desc]
    if not lines:
        return ""
    return (
        "COLUMN MEANINGS (from schema documentation — trust this over the "
        "field name alone):\n" + "\n".join(lines)
    )


def build_response_shaping_prompt(
    question: str,
    sql: str,
    rows: List[Dict],
    row_count: int,
    intent: Optional[str] = None,
    truncated: bool = False,
    include_few_shot: bool = True,
    max_sample_rows: Optional[int] = None,
    column_view: Optional[Dict[str, str]] = None,
) -> str:
    """
    Compose the full prompt sent to the response-shaping model.

    rows: the row SAMPLE already capped by the caller (gpu_response_service
          re-caps defensively via max_sample_rows too).
    row_count: the TRUE total row count (may be larger than len(rows)).
    column_view: optional {field_name: description} for fields present in
          `rows`, sourced from the module's narrow-json meta
          (gpu_response_service.py::_build_column_context). Purely additive
          context for the model — the hard PK-dropping backstop in
          gpu_response_service.py::_validate_and_repair does not depend on
          this and still enforces its own decision on the final field list
          regardless of what the model does with this information.
    """
    row_sample = _format_row_sample(rows, max_rows=max_sample_rows)
    column_view_block = _format_column_view(column_view)

    parts: List[str] = [RESPONSE_SHAPING_SYSTEM_RULES]
    if include_few_shot:
        parts.append(_FEW_SHOT_EXAMPLE_1)
        parts.append(_FEW_SHOT_EXAMPLE_2)

    intent_line = f"\nIntent hint: {intent}\n" if intent else ""
    truncated_line = (
        "\nNote: the full result set is larger than the sample shown here and was "
        "truncated for display — mention that in your summary if relevant.\n"
        if truncated else ""
    )
    column_view_line = f"\n{column_view_block}\n" if column_view_block else ""

    parts.append(
        "NOW DO THE REAL TASK.\n\n"
        f"Question: \"{question.strip()}\"\n"
        f"SQL: {sql.strip() if sql else '(not available)'}\n"
        f"Row count: {row_count} (truncated: {truncated})\n"
        f"{intent_line}{truncated_line}{column_view_line}\n"
        f"Sample rows:\n{row_sample}\n\n"
        "Output:"
    )
    return "\n\n".join(parts)