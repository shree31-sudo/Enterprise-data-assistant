#!/usr/bin/env python3
"""
MES Chatbot: Column Sensitivity Classifier
Extracts table/column metadata from mfc_narrow.json, sends to Qwen for classification.

Batched by SUBGROUP (e.g. BOM_MASTERS, BOM_STRUCTURE) rather than one giant
schema-wide call. This keeps each prompt/response small enough that Qwen
doesn't truncate output, and a failure in one subgroup doesn't lose the rest
of the run — results are saved incrementally after every subgroup.

Also supports RESUME: if column_sensitivity_classification.json already has
results for a subgroup's tables, that subgroup is skipped on re-run.
"""

import json
import requests
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configuration - Read from environment variables with defaults
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.225.30.26:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))  # per-subgroup call, not whole schema
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # Very low for deterministic JSON output
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "8000"))   # per-subgroup output cap
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384"))          # per-subgroup context window
MAX_RETRIES_PER_SUBGROUP = int(os.getenv("LLM_MAX_RETRIES", "2"))
RESUME = os.getenv("RESUME", "1") not in ("0", "false", "False")  # skip already-classified tables by default

# Paths - script is in core/ folder, data/ is at parent level
SCRIPT_DIR = Path(__file__).parent  # core/ folder
PROJECT_ROOT = SCRIPT_DIR.parent     # parent of core/
SOURCE_FILE = PROJECT_ROOT / "data" / "mfc_narrow.json"  # ../data/mfc_narrow.json
OUTPUT_FILE = SCRIPT_DIR / "column_sensitivity_classification.json"  # Save in core/ folder

SENSITIVITY_CLASSIFICATION_PROMPT = """You are a data sensitivity classifier for a Manufacturing Execution System (MES).

TASK: Classify each table and column as SENSITIVE or NOT SENSITIVE.

SENSITIVITY RULES (in order of priority):
1. HIGHLY SENSITIVE (mark as sensitive):
   - User personal data: Cuser, Uuser, user IDs, user emails, addresses, phone numbers, SSN, passport, salary
   - Audit trail columns: Cdate, Cuser, Udate, Uuser (internal tracking only)
   - Tenant_ID, Organization secrets, API keys, passwords, credentials
   - Employee personal details: salary, bank account, emergency contacts
   - Customer PII: email, phone, address, credit card, SSN
   - Manufacturing secrets: proprietary formulas, trade secrets, cost breakdowns, supplier agreements
   - Any column with "password", "secret", "token", "key", "credential" in name/description

2. MODERATELY SENSITIVE (mark as sensitive unless explicitly needed):
   - Internal IDs used only for joins: *_ID columns that are only foreign keys
   - Tenant/Organization data: internal org structures
   - Status codes that reveal internal workflows (unless business-critical)
   - Effective dates used only for data validity

3. NOT SENSITIVE (mark as not sensitive):
   - Public master data: Item names, BOM codes, product types, categories
   - Business operational columns: quantities, dates, status that users need to see
   - References needed for business decisions: part numbers, work order IDs
   - Descriptions and names of business entities
   - Date columns essential for reporting: effective dates, transaction dates
   - Status columns needed for decision-making

CONTEXT:
- MASTER tables typically contain reference data (less sensitive)
- LINK tables join reference data (check both sides)
- TRANSACTION tables record operations (timing/user fields sensitive, data fields not)

CRITICAL INSTRUCTIONS:
- OUTPUT ONLY VALID JSON
- NO MARKDOWN CODE FENCES (no ```json, no ```)
- NO PREAMBLE, NO EXPLANATION, NO TEXT BEFORE OR AFTER JSON
- START WITH { IMMEDIATELY
- CLASSIFY EVERY SINGLE TABLE AND EVERY SINGLE COLUMN GIVEN BELOW. DO NOT SKIP ANY.
- RESPOND WITH ONLY THIS JSON FORMAT:

{
  "table_name": {
    "sensitivity_assessment": "brief reason",
    "columns": {
      "column_name": {
        "sensitive": true/false,
        "reason": "why"
      }
    }
  }
}

NOW CLASSIFY THIS DATA AS JSON ONLY:
"""


class LLMCallError(Exception):
    """Raised when the LLM call itself fails (connection, timeout, bad status)."""
    pass


class LLMParseError(Exception):
    """Raised when the LLM response can't be parsed as JSON."""
    pass


def load_mfc_narrow(filepath) -> Dict[str, Any]:
    """Load mfc_narrow.json metadata file (accepts str or Path)"""
    filepath = Path(filepath)  # Convert to Path if string
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found")
        print(f"   Looked for: {filepath.absolute()}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{filepath}': {e}")
        sys.exit(1)


def load_existing_output(filepath) -> Dict[str, Any]:
    """Load prior classification output if present (for resume support). Returns {} if missing/invalid."""
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def build_column_context(table: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-table metadata dict (table_type, business_purpose, columns) used in prompts."""
    table_type = table.get('table_type')
    business_purpose = table.get('business_purpose', '')
    columns = table.get('columns', {})
    time_columns = table.get('time_columns', [])
    status_columns = table.get('status_columns', [])

    columns_with_context = {}
    for col_name, col_description in columns.items():
        columns_with_context[col_name] = {
            "description": col_description,
            "is_time_column": col_name in time_columns,
            "is_status_column": col_name in status_columns
        }

    return {
        "table_type": table_type,
        "business_purpose": business_purpose,
        "columns": columns_with_context
    }


def extract_subgroups(mfc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Traverse modules -> subgroups -> tables and return a flat list of subgroup
    batches, each ready to be classified independently:

    [
      {
        "module_key": "ENGINEERING",
        "subgroup_key": "BOM_MASTERS",
        "tables": { table_name: {table_type, business_purpose, columns}, ... }
      },
      ...
    ]
    """
    subgroups = []

    for module_key, module_data in mfc_data.get('modules', {}).items():
        for subgroup_key, subgroup_data in module_data.get('subgroups', {}).items():
            tables = {}
            for table in subgroup_data.get('tables', []):
                table_name = table.get('table_name')
                if not table_name:
                    continue
                tables[table_name] = build_column_context(table)

            if tables:
                subgroups.append({
                    "module_key": module_key,
                    "subgroup_key": subgroup_key,
                    "tables": tables
                })

    return subgroups


def call_qwen_ollama(prompt: str) -> str:
    """Call Qwen via Ollama and return the raw response text. Raises LLMCallError on failure."""
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,  # Ollama's actual field to disable Qwen3 thinking mode
                "format": "json",  # Force Ollama's JSON-constrained output mode
                "options": {
                    "temperature": LLM_TEMPERATURE,  # Must be nested under "options", not top-level
                    "num_predict": LLM_NUM_PREDICT,
                    "num_ctx": LLM_NUM_CTX
                }
            },
            timeout=(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT)
        )

        if response.status_code != 200:
            raise LLMCallError(f"LLM returned status {response.status_code}: {response.text[:500]}")

        result = response.json()
        return result.get('response', '')

    except requests.exceptions.ConnectionError:
        raise LLMCallError(f"Cannot connect to LLM at {LLM_BASE_URL}. Make sure Ollama is running: ollama serve")
    except requests.exceptions.Timeout:
        raise LLMCallError(f"Request timed out (connect: {LLM_CONNECT_TIMEOUT}s, read: {LLM_TIMEOUT}s)")
    except LLMCallError:
        raise
    except Exception as e:
        raise LLMCallError(f"Unexpected error calling Ollama: {e}")


def parse_qwen_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from Qwen response. Handles markdown fences, leading/trailing text.
    Raises LLMParseError on failure.
    """
    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    if not cleaned.startswith("{"):
        start_idx = cleaned.find("{")
        if start_idx != -1:
            cleaned = cleaned[start_idx:]

    if not cleaned.endswith("}"):
        end_idx = cleaned.rfind("}")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"Could not parse response as JSON: {e}\n"
            f"Raw response (first 500 chars): {response_text[:500]}"
        )


def validate_classification(classification: Dict[str, Any], context: str = "") -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Validate structure of parsed classification results.
    Qwen occasionally deviates from the requested schema (e.g. returns a
    plain string instead of a table dict). Drop malformed entries rather
    than crashing downstream. Returns (validated_dict, dropped_list).
    """
    validated = {}
    dropped = []

    for table_name, table_data in classification.items():
        if not isinstance(table_data, dict):
            dropped.append((f"{context}{table_name}", f"table value is {type(table_data).__name__}, expected dict"))
            continue

        columns = table_data.get('columns', {})
        if not isinstance(columns, dict):
            dropped.append((f"{context}{table_name}", f"'columns' is {type(columns).__name__}, expected dict"))
            continue

        clean_columns = {}
        for col_name, col_data in columns.items():
            if not isinstance(col_data, dict) or 'sensitive' not in col_data:
                dropped.append((f"{context}{table_name}.{col_name}", "malformed column entry, skipped"))
                continue
            clean_columns[col_name] = col_data

        table_data['columns'] = clean_columns
        validated[table_name] = table_data

    return validated, dropped


def classify_subgroup(subgroup: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Classify a single subgroup's tables. Retries on call/parse failure up to
    MAX_RETRIES_PER_SUBGROUP times. Returns (validated_results, dropped_entries).
    Raises the last exception if all attempts fail.
    """
    input_json_str = json.dumps(subgroup["tables"], indent=2)
    full_prompt = SENSITIVITY_CLASSIFICATION_PROMPT + "\n\n" + input_json_str

    last_error = None
    for attempt in range(1, MAX_RETRIES_PER_SUBGROUP + 2):  # e.g. retries=2 -> 3 attempts total
        try:
            response_text = call_qwen_ollama(full_prompt)
            parsed = parse_qwen_response(response_text)
            validated, dropped = validate_classification(parsed, context=f"{subgroup['subgroup_key']}.")
            return validated, dropped
        except (LLMCallError, LLMParseError) as e:
            last_error = e
            print(f"      ⚠️  Attempt {attempt} failed: {e}")
            if attempt <= MAX_RETRIES_PER_SUBGROUP:
                time.sleep(2)

    raise last_error


def save_output(data: Dict[str, Any], filepath) -> None:
    """Save classification results to JSON file (accepts str or Path)"""
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving output: {e}")


def print_summary(classification: Dict[str, Any], failed_subgroups: List[str], all_dropped: List[Tuple[str, str]]) -> None:
    """Print summary of classification results"""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    total_tables = len(classification)
    total_columns = 0
    sensitive_columns = 0

    for table_name, table_data in classification.items():
        columns = table_data.get('columns', {})
        total_columns += len(columns)
        for col_data in columns.values():
            if col_data.get('sensitive', False):
                sensitive_columns += 1

    print(f"\n📊 Statistics:")
    print(f"   Tables classified: {total_tables}")
    print(f"   Total Columns: {total_columns}")
    print(f"   Sensitive Columns: {sensitive_columns}")
    print(f"   Non-Sensitive Columns: {total_columns - sensitive_columns}")
    if total_columns > 0:
        print(f"   Sensitivity Rate: {(sensitive_columns/total_columns*100):.1f}%")
    else:
        print(f"   Sensitivity Rate: N/A (no valid columns)")

    if failed_subgroups:
        print(f"\n❌ Subgroups that failed after retries ({len(failed_subgroups)}):")
        for sg in failed_subgroups:
            print(f"   • {sg}")
        print("   Re-run the script — RESUME=1 (default) will skip already-completed")
        print("   subgroups and retry only the failed ones.")

    if all_dropped:
        print(f"\n⚠️  {len(all_dropped)} malformed column/table entries dropped across all subgroups:")
        for name, reason in all_dropped[:20]:
            print(f"   • {name}: {reason}")
        if len(all_dropped) > 20:
            print(f"   ... and {len(all_dropped) - 20} more")

    print("\n" + "=" * 80)


def main():
    """Main execution flow"""
    print("\n" + "=" * 80)
    print("MES CHATBOT: COLUMN SENSITIVITY CLASSIFIER (batched by subgroup)")
    print("=" * 80)

    # Step 1: Load mfc_narrow.json metadata
    print(f"\n📂 Step 1: Loading metadata from '{SOURCE_FILE}'...")
    mfc_data = load_mfc_narrow(SOURCE_FILE)
    print(f"   ✅ Loaded mfc_narrow.json")

    # Step 2: Extract subgroups (e.g. ENGINEERING.BOM_MASTERS, ENGINEERING.BOM_STRUCTURE, ...)
    print(f"\n🔍 Step 2: Extracting subgroups...")
    subgroups = extract_subgroups(mfc_data)
    total_tables = sum(len(sg["tables"]) for sg in subgroups)
    total_cols = sum(len(t.get('columns', {})) for sg in subgroups for t in sg["tables"].values())
    print(f"   ✅ Found {len(subgroups)} subgroups, {total_tables} tables, {total_cols} columns")

    # Step 2b: Load existing results for resume support
    classification_results: Dict[str, Any] = load_existing_output(OUTPUT_FILE) if RESUME else {}
    if classification_results:
        print(f"   ↩️  RESUME enabled: found {len(classification_results)} already-classified tables in {OUTPUT_FILE.name}")

    # Step 3: Classify each subgroup independently, saving after every one
    failed_subgroups: List[str] = []
    all_dropped: List[Tuple[str, str]] = []
    skipped_subgroups = 0

    for i, subgroup in enumerate(subgroups, 1):
        label = f"{subgroup['module_key']}.{subgroup['subgroup_key']}"
        n_tables = len(subgroup["tables"])
        n_cols = sum(len(t.get('columns', {})) for t in subgroup["tables"].values())

        # Resume: skip subgroup only if ALL its tables already have valid results
        if RESUME and all(t in classification_results for t in subgroup["tables"]):
            skipped_subgroups += 1
            print(f"\n⏭️  [{i}/{len(subgroups)}] Skipping '{label}' — already classified ({n_tables} tables)")
            continue

        print(f"\n🤖 [{i}/{len(subgroups)}] Classifying subgroup '{label}' ({n_tables} tables, {n_cols} columns)...")

        try:
            validated, dropped = classify_subgroup(subgroup)
            classification_results.update(validated)
            all_dropped.extend(dropped)
            print(f"      ✅ Classified {len(validated)} tables ({len(dropped)} entries dropped)")
        except (LLMCallError, LLMParseError) as e:
            print(f"      ❌ Subgroup '{label}' failed after {MAX_RETRIES_PER_SUBGROUP + 1} attempts: {e}")
            failed_subgroups.append(label)

        # Save incrementally so a crash/interrupt doesn't lose completed subgroups
        save_output(classification_results, OUTPUT_FILE)

    if skipped_subgroups:
        print(f"\n↩️  Skipped {skipped_subgroups} already-classified subgroup(s) (RESUME=1)")

    print(f"\n💾 Final save: {OUTPUT_FILE}")
    save_output(classification_results, OUTPUT_FILE)
    print(f"   ✅ Saved {len(classification_results)} tables total")

    # Step 4: Summary
    print_summary(classification_results, failed_subgroups, all_dropped)

    print(f"\n✨ Classification complete!")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   Use this file to filter columns in ResponseTranslator")
    if failed_subgroups:
        print(f"   ⚠️  {len(failed_subgroups)} subgroup(s) failed — re-run to retry them (RESUME=1 skips completed ones)")
        sys.exit(1)
    print("\n")


if __name__ == "__main__":
    main()