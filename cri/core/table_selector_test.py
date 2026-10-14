
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ============================================================
# CONFIG — adjust here or via env vars if paths/model differ
# ============================================================
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
JOIN_GRAPH_DIR = Path(os.getenv("JOIN_GRAPH_DIR", str(DATA_DIR / "join_graphs"))).resolve()
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://10.225.30.26:11434/api/generate").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5:3b").strip()
LLM_TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "60"))

# Same intent vocabulary already used elsewhere in the pipeline, so qwen's
# output stays compatible with the rest of the system if we wire this in later.
INTENT_LABELS = [
    "simple", "lookup", "aggregation", "group_by", "sort_by",
    "existence", "conditional", "time_based", "top_bottom", "search",
    "relational_lookup", "relational_aggregation", "relational",
]

MAX_TABLE_CATALOG_CHARS = 6000   # keep prompt bounded even for big modules
MAX_COLS_PER_TABLE_HINT = 12     # trimmed column preview per table (if present)


# ============================================================
# STEP 1 — discover modules on disk
# ============================================================
def discover_modules(data_dir: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    """
    Returns: { module_key: {"narrow": Path or None, "join": Path or None} }

    module_key is derived from the filename prefix before '_narrow.json'.
    Join file is matched by same prefix + anything containing 'join'.
    """
    modules: Dict[str, Dict[str, Optional[Path]]] = {}

    for narrow_path in sorted(glob.glob(str(data_dir / "*_narrow.json"))):
        p = Path(narrow_path)
        module_key = p.name[: -len("_narrow.json")]
        modules.setdefault(module_key, {"narrow": None, "join": None})
        modules[module_key]["narrow"] = p

    for module_key in list(modules.keys()):
        # Join graph files live in their own subfolder, named <module>_join_graph.json
        # (module_key here is upper/lower-cased as it appears in the *_narrow.json
        # filename; we try both cases since prefixes can be mixed-case on disk).
        join_candidates = sorted(
            glob.glob(str(JOIN_GRAPH_DIR / f"{module_key}_join_graph.json"))
            + glob.glob(str(JOIN_GRAPH_DIR / f"{module_key.lower()}_join_graph.json"))
            + glob.glob(str(JOIN_GRAPH_DIR / f"{module_key.upper()}_join_graph.json"))
        )
        if join_candidates:
            modules[module_key]["join"] = Path(join_candidates[0])

    return modules


# ============================================================
# STEP 2 — parse a module's narrow.json into a flat table catalog
# ============================================================
def load_table_catalog(narrow_path: Path) -> List[dict]:
    """
    Mirrors the structure used in chatbot_backend.py's _build_narrow_table_index:
        { "modules": { mod: { "subgroups": { sg: { "tables": [ {...} ] } } } } }
    Falls back gracefully if a module/subgroup entry is a bare list instead
    of a dict with a "tables" key.
    """
    raw = json.loads(narrow_path.read_text(encoding="utf-8"))
    catalog: List[dict] = []

    for mod_val in raw.get("modules", {}).values():
        for sg_val in mod_val.get("subgroups", {}).values():
            sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
            for tbl in sg_list:
                tname = tbl.get("table_name", "")
                if tname:
                    catalog.append(tbl)

    return catalog


def format_table_catalog(catalog: List[dict]) -> str:
    """
    Turns the raw table dicts into a compact, LLM-readable block:
        - TABLE_NAME: <description if present> | parent: X | cols: a, b, c...
    Doesn't assume a fixed schema — pulls whatever descriptive fields exist.
    """
    lines: List[str] = []
    for tbl in catalog:
        name = tbl.get("table_name", "")
        if not name:
            continue

        bits: List[str] = []

        desc = tbl.get("description") or tbl.get("desc") or tbl.get("purpose")
        if desc:
            bits.append(str(desc))

        parent = tbl.get("parent_entity")
        if parent:
            bits.append(f"parent: {parent}")

        cols = tbl.get("columns") or tbl.get("business_columns")
        if cols:
            preview = ", ".join(cols[:MAX_COLS_PER_TABLE_HINT])
            more = f" (+{len(cols) - MAX_COLS_PER_TABLE_HINT} more)" if len(cols) > MAX_COLS_PER_TABLE_HINT else ""
            bits.append(f"cols: {preview}{more}")

        line = f"- {name}" + (f": {' | '.join(bits)}" if bits else "")
        lines.append(line)

    text = "\n".join(lines)
    if len(text) > MAX_TABLE_CATALOG_CHARS:
        text = text[:MAX_TABLE_CATALOG_CHARS] + "\n... (truncated)"
    return text


# ============================================================
# STEP 3 — parse a module's join graph into readable edges
# ============================================================
def load_join_edges(join_path: Optional[Path]) -> str:
    """
    Real shape of <module>_join_graph.json (per dl_join_graph.json sample):
        {
          "_meta": {...},
          "graph": { "TableA": ["TableB", ...], "TableB": [...] },
          "join_paths": {
            "TableA": {
              "TableB": [
                {"from_table": "TableA", "from_column": "X",
                 "to_table": "TableB",   "to_column": "X"}
              ]
            }
          }
        }

    We render join_paths (it has the actual join columns) as:
        TableA.from_column = TableB.to_column
    and fall back to the plain "graph" adjacency (no column info) if
    join_paths is missing. If the file shape is something else entirely,
    we just pretty-print the raw JSON so the model still gets *something*.
    """
    if join_path is None or not join_path.exists():
        return "(no join graph file found for this module)"

    raw = json.loads(join_path.read_text(encoding="utf-8"))
    lines: List[str] = []
    seen: set = set()

    join_paths = raw.get("join_paths") if isinstance(raw, dict) else None
    if isinstance(join_paths, dict):
        for base_tbl, targets in join_paths.items():
            if not isinstance(targets, dict):
                continue
            for tgt_tbl, conditions in targets.items():
                if not isinstance(conditions, list):
                    continue
                for cond in conditions:
                    ft = cond.get("from_table", base_tbl)
                    fc = cond.get("from_column", "?")
                    tt = cond.get("to_table", tgt_tbl)
                    tc = cond.get("to_column", "?")
                    line = f"{ft}.{fc} = {tt}.{tc}"
                    key = tuple(sorted((f"{ft}.{fc}", f"{tt}.{tc}")))
                    if key not in seen:
                        seen.add(key)
                        lines.append(line)

    if not lines:
        graph = raw.get("graph") if isinstance(raw, dict) else None
        if isinstance(graph, dict):
            for base_tbl, targets in graph.items():
                for tgt_tbl in targets:
                    key = tuple(sorted((base_tbl, tgt_tbl)))
                    if key not in seen:
                        seen.add(key)
                        lines.append(f"{base_tbl} -> {tgt_tbl} (join column unknown)")

    if not lines:
        dumped = json.dumps(raw, indent=2)
        return dumped[:MAX_TABLE_CATALOG_CHARS]

    text = "\n".join(lines)
    if len(text) > MAX_TABLE_CATALOG_CHARS:
        text = text[:MAX_TABLE_CATALOG_CHARS] + "\n... (truncated)"
    return text


# ============================================================
# STEP 4 — build the qwen prompt
# ============================================================
def build_prompt(question: str, module_key: str, table_catalog_text: str, join_edges_text: str) -> str:
    return f"""You are a table-selection assistant for a natural-language-to-SQL system.

Module: {module_key}

Available tables in this module:
{table_catalog_text}

Known join relationships between tables in this module:
{join_edges_text}

User question:
"{question}"

Task:
1. Decide the query intent — pick exactly one from this list:
   {", ".join(INTENT_LABELS)}
2. Decide the final set of table(s) needed to answer the question.
   - Pick only tables that are actually needed.
   - If more than one table is needed, only include tables that have a
     known join relationship to at least one other selected table.
   - Prefer the smallest table set that fully answers the question.

Respond with ONLY a JSON object, no explanation, no markdown fences, in exactly this shape:
{{"intent": "<one of the intent labels>", "tables": ["TABLE_ONE", "TABLE_TWO"]}}
"""


# ============================================================
# STEP 5 — call qwen via Ollama
# ============================================================
def call_qwen(prompt: str) -> str:
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 200,
            "temperature": 0.0,
        },
    }
    resp = requests.post(LLM_SERVER_URL, json=payload, timeout=LLM_TIMEOUT_SECS)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def parse_model_output(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ============================================================
# MAIN — interactive loop
# ============================================================
def main() -> None:
    print(f"\nScanning {DATA_DIR} for '*_narrow.json' modules ...\n")

    modules = discover_modules(DATA_DIR)
    if not modules:
        print(f"No '*_narrow.json' files found under {DATA_DIR}.")
        print("Set DATA_DIR env var to point at the folder containing them, e.g.:")
        print(f"    DATA_DIR=/path/to/data python {Path(__file__).name}")
        sys.exit(1)

    module_keys = sorted(modules.keys())
    print("Discovered modules:")
    for i, mk in enumerate(module_keys, 1):
        narrow_ok = "✓" if modules[mk]["narrow"] else "✗"
        join_ok = "✓" if modules[mk]["join"] else "✗ (no join file matched)"
        print(f"  {i}. {mk}   [narrow: {narrow_ok}]  [join: {join_ok}]")

    while True:
        choice = input("\nPick a module by number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(module_keys)):
            print("Invalid choice.")
            continue
        module_key = module_keys[int(choice) - 1]
        break

    narrow_path = modules[module_key]["narrow"]
    join_path = modules[module_key]["join"]

    catalog = load_table_catalog(narrow_path)
    if not catalog:
        print(f"Warning: no tables parsed out of {narrow_path}. Check its structure.")
    table_catalog_text = format_table_catalog(catalog)
    join_edges_text = load_join_edges(join_path)

    print(f"\nLoaded {len(catalog)} table(s) for module '{module_key}'.")
    print(f"LLM: {QWEN_MODEL} @ {LLM_SERVER_URL}")
    print("\nType a question below. Type 'q' to quit, 'm' to switch module.\n")

    while True:
        question = input(f"[{module_key}] question> ").strip()
        if not question:
            continue
        if question.lower() == "q":
            return
        if question.lower() == "m":
            return main()

        prompt = build_prompt(question, module_key, table_catalog_text, join_edges_text)

        try:
            raw = call_qwen(prompt)
        except requests.exceptions.RequestException as exc:
            print(f"  ✗ LLM call failed: {exc}")
            continue

        print("\n  --- raw model output ---")
        print(f"  {raw}")

        parsed = parse_model_output(raw)
        if parsed is None:
            print("  ✗ Could not parse JSON from model output.\n")
            continue

        intent = parsed.get("intent", "<missing>")
        tables = parsed.get("tables", [])

        print("\n  --- parsed result ---")
        print(f"  intent : {intent}")
        print(f"  tables : {tables}\n")


if __name__ == "__main__":
    main()
