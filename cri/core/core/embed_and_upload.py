from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embed_upload")

# ── project root ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# ── constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
DEFAULT_SCHEMA_JSON = str(_PROJECT_ROOT / "data" / "schema_raw.json")

# ── Module config — driven by TABLE_PREFIX in .env ────────────────────────────
_TABLE_PREFIX_RAW: str = os.getenv("TABLE_PREFIX", "MFC")
_MODULE_LIST: List[str] = [m.strip().upper() for m in _TABLE_PREFIX_RAW.split(",") if m.strip()]
_TABLE_NAME_PREFIXES_RAW: str = os.getenv("TABLE_NAME_PREFIXES", "MFC_M_,MFC_L_,MFC_T_")
_TABLE_NAME_PREFIXES: List[str] = [p.strip() for p in _TABLE_NAME_PREFIXES_RAW.split(",") if p.strip()]


_COMBINED_NARROW = _PROJECT_ROOT / "data" / "prod_narrow.json"
if _COMBINED_NARROW.exists():
    _ALL_NARROW_JSONS: List[str] = [str(_COMBINED_NARROW)]
else:
    _ALL_NARROW_JSONS = [
        str(_PROJECT_ROOT / "data" / f"{m.lower()}_narrow.json")
        for m in _MODULE_LIST
    ]
# Default for --narrow-json arg: comma-joined list of all module files
DEFAULT_NARROW_JSON = ",".join(_ALL_NARROW_JSONS)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384").strip())

log.info(f"[embed_upload] model={MODEL_NAME!r}  dim={VECTOR_DIM}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEXT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_mfc_text(module_name: str, subgroup_name: str, table: dict) -> str:
    """Table‑level text from mfc_narrow.json (now includes all metadata)."""
    lines = [
        f"module: {module_name}",
        f"subgroup: {subgroup_name}",
        f"table: {table['table_name']}"
    ]
    purpose = table.get("business_purpose", "").strip()
    if purpose:
        lines.append(f"purpose: {purpose}")
    row_repr = table.get("row_represents", "").strip()
    if row_repr:
        lines.append(f"row represents: {row_repr}")
    synonyms = table.get("synonyms", [])
    if synonyms:
        lines.append("synonyms: " + ", ".join(synonyms))
    use_when = table.get("use_when", [])
    if use_when:
        lines.append("use when: " + "; ".join(use_when))
    # --- new fields for advanced selection logic ---
    query_patterns = table.get("query_patterns", [])
    if query_patterns:
        lines.append("query patterns: " + "; ".join(query_patterns))
    do_not_use = table.get("do_not_use_when", [])
    if do_not_use:
        lines.append("do not use when: " + "; ".join(do_not_use))
    table_type = table.get("table_type", "")
    if table_type:
        lines.append(f"table type: {table_type}")
    routing_hint = table.get("model_routing_hint", "")
    if routing_hint:
        lines.append(f"model routing hint: {routing_hint}")
    # --- end new fields ---
    columns = table.get("columns", {})
    if columns:
        lines.append("columns: " + ", ".join(columns.keys()))
        col_descs = [v.strip() for v in columns.values() if v and v.strip()]
        if col_descs:
            lines.append("column descriptions: " + "; ".join(col_descs))
    return "\n".join(lines)


def build_column_text(table_name: str, col: dict) -> str:
    """
    One string per column with semantically rich embedding text.

    Synthesises all available metadata — description, business purpose,
    row context, synonyms, and use-when guidance — into natural-language
    sentences rather than generic "data type: X" labels.  The goal is an
    embedding that captures *what the column means and when to use it*,
    not just its technical type.

    The col dict is expected to come from schema_raw.json and may contain:
        column_name, data_type, semantic_role, meaning,
        description, business_purpose, row_represents,
        synonyms, use_when, sample_values
    All fields are optional; missing ones are silently skipped.
    """
    col_name    = col.get("column_name", "")
    semantic    = col.get("semantic_role", "").strip()
    meaning     = col.get("meaning", "").strip()
    description = col.get("description", "").strip()
    biz_purpose = col.get("business_purpose", "").strip()
    row_repr    = col.get("row_represents", "").strip()
    synonyms    = col.get("synonyms", [])
    use_when    = col.get("use_when", [])
    sample_vals = col.get("sample_values", [])

    # ── Human-readable column label (strip MFC prefix noise) ──────────────
    human_col = col_name.replace("_", " ").strip()

    # ── Lead sentence: what this column IS ────────────────────────────────
    # Priority: description > meaning > semantic_role (fallback to col name)
    primary_desc = description or meaning or semantic
    if primary_desc:
        lead = f"{human_col} in {table_name}: {primary_desc}."
    else:
        lead = f"{human_col} in {table_name}."

    parts = [lead]

    # ── Business purpose ───────────────────────────────────────────────────
    # Avoid repeating identical text already in the lead sentence
    if biz_purpose and biz_purpose not in primary_desc:
        parts.append(f"Business purpose: {biz_purpose}.")

    # ── Row context (what one row represents) ──────────────────────────────
    if row_repr:
        parts.append(f"Each row represents: {row_repr}.")

    # ── Synonyms / alternate names ─────────────────────────────────────────
    if synonyms:
        syn_str = ", ".join(str(s) for s in synonyms)
        parts.append(f"Also known as: {syn_str}.")

    # ── Use-when guidance (most valuable for retrieval intent matching) ────
    if use_when:
        uw_str = "; ".join(str(u) for u in use_when)
        parts.append(f"Use this column when: {uw_str}.")

    # ── Sample values (grounding for numeric / code columns) ──────────────
    if sample_vals:
        sv_str = ", ".join(str(v) for v in sample_vals[:5])
        parts.append(f"Example values: {sv_str}.")

    return " ".join(parts)

def build_entity_text(entity: Dict[str, str]) -> str:
    """
    Phrase-level semantic entity embedding text.
    """
    return "\n".join([
        f"phrase: {entity['phrase']}",
        f"table: {entity['table_name']}",
        f"source: {entity['source']}"
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXTRACT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_table_list(sg_val) -> list:
    """
    Resolve a subgroup value to a flat list of table dicts.

    Supports two formats:
      v1 (legacy): sg_val is a bare list of table dicts
      v2 (current): sg_val is a dict with keys
                    {"scope_hard_block_keywords": [...], "tables": [...]}
    """
    if isinstance(sg_val, dict):
        return sg_val.get("tables", [])
    if isinstance(sg_val, list):
        return sg_val
    return []


def extract_mfc_tables(json_path: str) -> List[Tuple[str, str, dict]]:
    """
    Extract all (module_name, subgroup_name, table_dict) triples from a
    narrow JSON file.

    Supports two file shapes:
      FLAT (current, e.g. prod_narrow.json):
          {"module_id": "PROD", "module_label": "...", "tables": [ {...}, ... ]}
      NESTED (legacy, per-prefix files):
          {"modules": {mod_name: {"subgroups": {sg_name: {"tables": [...]}}}}}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []

    if "tables" in data and isinstance(data["tables"], list):
        # FLAT shape — one module, one subgroup for every table
        module_name = data.get("module_id") or data.get("module_label") or "PROD"
        subgroup_name = "production_operations"
        for table in data["tables"]:
            records.append((module_name, subgroup_name, table))
    else:
        # NESTED shape — original per-module/per-subgroup structure
        for module_name, module in data.get("modules", {}).items():
            for subgroup_name, sg_val in module.get("subgroups", {}).items():
                # FIX: resolve v2 subgroup format {"scope_hard_block_keywords": [...], "tables": [...]}
                for table in _get_table_list(sg_val):
                    records.append((module_name, subgroup_name, table))

    log.info(f"Extracted {len(records)} tables from {json_path}")
    return records


def extract_schema_columns(json_path: str) -> List[Tuple[str, dict]]:
    """
    Extract all columns from schema_raw.json.
    Returns:
        List of (table_name, column_dict)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for table_name, table_meta in data.items():
        columns = table_meta.get("columns", [])
        for col in columns:
            records.append((table_name, col))

    log.info(f"Extracted {len(records)} schema columns from {json_path}")
    return records


def extract_semantic_entities(json_path: str) -> List[Dict[str, str]]:
    """
    Build phrase-level semantic entity records from metadata.
    Handles both v1 (bare list) and v2 (dict with 'tables' key)
    subgroup formats.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    seen = set()

    def add_phrase(phrase: str, table_name: str, source: str):
        p = phrase.strip().lower()
        if not p or len(p.split()) < 2:
            return
        key = (p, table_name)
        if key in seen:
            return
        seen.add(key)
        records.append({
            "phrase": p,
            "table_name": table_name,
            "source": source
        })

    if "tables" in data and isinstance(data["tables"], list):
        table_iter = data["tables"]
    else:
        table_iter = [
            table
            for module in data.get("modules", {}).values()
            for sg_val in module.get("subgroups", {}).values()
            # FIX: resolve v2 subgroup format {"scope_hard_block_keywords": [...], "tables": [...]}
            for table in _get_table_list(sg_val)
        ]

    for table in table_iter:
        table_name = table["table_name"]

        # table name phrase — strip all configured prefixes dynamically
        human_name = table_name
        for _pfx in _TABLE_NAME_PREFIXES:
            human_name = human_name.replace(_pfx, "")
        human_name = human_name.replace("_", " ").lower()

        if len(human_name.split()) >= 2:
            add_phrase(human_name, table_name, "table_name")

        # synonym phrases
        for s in table.get("synonyms", []):
            add_phrase(s, table_name, "synonym")

        # use_when phrases
        for s in table.get("use_when", []):
            add_phrase(s, table_name, "use_when")

        # do_not_use_when phrases
        for s in table.get("do_not_use_when", []):
            add_phrase(s, table_name, "do_not_use_when")

    log.info(f"Extracted {len(records)} semantic entity phrases")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# 3.  QDRANT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        log.error("QDRANT_URL and QDRANT_API_KEY must be set in .env or environment.")
        sys.exit(1)
    log.info(f"Connecting to Qdrant at {url} …")
    client = QdrantClient(url=url, api_key=api_key, timeout=60)
    log.info("Qdrant connection established.")
    return client


def ensure_collection(client: QdrantClient, collection_name: str, recreate: bool = False) -> None:
    exists = any(c.name == collection_name for c in client.get_collections().collections)
    if exists and recreate:
        log.warning(f"--recreate flag set — deleting collection '{collection_name}'")
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        log.info(f"Creating collection '{collection_name}' (dim={VECTOR_DIM}, cosine) …")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )
        # Index common filter fields
        for field in ("table_name", "column_name", "source", "phrase"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            log.info(f"  Payload index created: {field}")
    else:
        log.info(f"Collection '{collection_name}' already exists — upserting.")


def stable_point_id_mfc(module: str, subgroup: str, table_name: str) -> str:
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    key = f"mfc::{module}::{subgroup}::{table_name}"
    return str(uuid.uuid5(namespace, key))


def stable_point_id_schema_column(table_name: str, column_name: str) -> str:
    """Deterministic ID for a column in schema_raw."""
    namespace = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
    key = f"schema_column::{table_name}::{column_name}"
    return str(uuid.uuid5(namespace, key))

def stable_point_id_entity(phrase: str, table_name: str) -> str:
    namespace = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")
    key = f"semantic_entity::{table_name}::{phrase}"
    return str(uuid.uuid5(namespace, key))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Embed module narrow JSONs (table-level) and/or schema_raw.json (column-level) into Qdrant. "
            "Modules are driven by TABLE_PREFIX in .env."
        )
    )
    parser.add_argument(
        "--source", choices=["mfc", "schema", "entities", "both"], default="both",
        help="Which JSON source to embed (default: both). 'mfc' embeds all module narrow JSONs."
    )
    parser.add_argument(
        "--narrow-json", default=DEFAULT_NARROW_JSON,
        help=(
            "Comma-separated paths to module narrow JSON files. "
            f"Defaults to all modules in TABLE_PREFIX: {DEFAULT_NARROW_JSON}"
        ),
    )
    parser.add_argument(
        "--schema-json", default=DEFAULT_SCHEMA_JSON,
        help="Path to schema_raw.json"
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the target collections (dangerous)"
    )
    args = parser.parse_args()

    # Resolve narrow JSON file list
    narrow_json_paths: List[str] = [
        p.strip() for p in args.narrow_json.split(",") if p.strip()
    ]

    client = get_qdrant_client()
    model = SentenceTransformer(MODEL_NAME)
    log.info(f"Loaded embedding model: {MODEL_NAME}")
    log.info(f"Modules to embed: {_MODULE_LIST}")

    # ---- Process all module narrow JSONs (table-level) ----
    if args.source in ("mfc", "both"):
        # ── Group records by module and upsert into per-module collections ──
        # Each narrow JSON file is named <module>_narrow.json, so we derive
        # the collection name from the source file rather than hardcoding "mfc_tables".
        from collections import defaultdict
        per_module: Dict[str, list] = defaultdict(list)  # module_lower → [(text, payload, point_id)]
        for narrow_json in narrow_json_paths:
            if not Path(narrow_json).exists():
                continue
            module_key = Path(narrow_json).stem.replace("_narrow", "").lower()  # e.g. "mfc", "usr", "alm"
            records_for_file = extract_mfc_tables(narrow_json)
            for module, subgroup, table in records_for_file:
                text = build_mfc_text(module, subgroup, table)
                payload = {
                    "table_name": table["table_name"],
                    "table_type": table.get("table_type", ""),
                    "module": module,
                    "subgroup": subgroup,
                    "source": "mfc_narrow",
                    "business_purpose": table.get("business_purpose", ""),
                    "row_represents": table.get("row_represents", ""),
                    "synonyms": table.get("synonyms", []),
                    "use_when": table.get("use_when", []),
                    "query_patterns": table.get("query_patterns", []),
                    "do_not_use_when": table.get("do_not_use_when", []),
                    "model_routing_hint": table.get("model_routing_hint", ""),
                }
                pid = stable_point_id_mfc(module, subgroup, table["table_name"])
                per_module[module_key].append((text, payload, pid))

        for module_key, entries in per_module.items():
            coll_name = f"{module_key}_tables"   # e.g. mfc_tables, usr_tables, alm_tables
            texts     = [e[0] for e in entries]
            payloads  = [e[1] for e in entries]
            point_ids = [e[2] for e in entries]

            log.info(f"Embedding {len(texts)} tables → collection '{coll_name}' …")
            vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True).tolist()
            if vectors and len(vectors[0]) != VECTOR_DIM:
                log.error(f"Dimension mismatch: model produced {len(vectors[0])} but VECTOR_DIM={VECTOR_DIM}. Check .env.")
                sys.exit(1)

            ensure_collection(client, coll_name, recreate=args.recreate)
            for i in range(0, len(vectors), BATCH_SIZE):
                batch_points = [
                    qmodels.PointStruct(
                        id=point_ids[j],
                        vector=vectors[j],
                        payload=payloads[j],
                    )
                    for j in range(i, min(i + BATCH_SIZE, len(vectors)))
                ]
                client.upsert(collection_name=coll_name, points=batch_points)
                log.info(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)} to '{coll_name}'")

    # ---- Process schema_raw.json (column‑level) ----
    if args.source in ("schema", "both"):
        if not Path(args.schema_json).exists():
            log.error(f"schema JSON file not found: {args.schema_json}")
            sys.exit(1)

        records = extract_schema_columns(args.schema_json)
        texts, payloads, point_ids = [], [], []
        for table_name, col in records:
            col_name = col.get("column_name", "")
            texts.append(build_column_text(table_name, col))
            payloads.append({
                "table_name": table_name,
                "column_name": col_name,
                "data_type": col.get("data_type", ""),
                "semantic_role": col.get("semantic_role", ""),
                "meaning": col.get("meaning", ""),
                "sample_values": col.get("sample_values", []),
                "source": "schema_raw"
            })
            point_ids.append(stable_point_id_schema_column(table_name, col_name))

        if texts:
            vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True).tolist()
            if vectors and len(vectors[0]) != VECTOR_DIM:
                log.error(f"Dimension mismatch: model produced {len(vectors[0])} but VECTOR_DIM={VECTOR_DIM}. Check .env.")
                sys.exit(1)

            coll_name = "schema_columns"
            ensure_collection(client, coll_name, recreate=args.recreate)
            for i in range(0, len(vectors), BATCH_SIZE):
                batch_points = [
                    qmodels.PointStruct(
                        id=point_ids[j],
                        vector=vectors[j],
                        payload=payloads[j],
                    )
                    for j in range(i, min(i + BATCH_SIZE, len(vectors)))
                ]
                client.upsert(collection_name=coll_name, points=batch_points)
                log.info(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)} to {coll_name}")

    # ---- Process semantic entity phrases (all modules) ----
    if args.source in ("entities", "both"):
        records = []
        for narrow_json in narrow_json_paths:
            if not Path(narrow_json).exists():
                log.warning(f"Narrow JSON not found for entities, skipping: {narrow_json}")
                continue
            records.extend(extract_semantic_entities(narrow_json))

        texts = []
        payloads = []
        point_ids = []

        for entity in records:

            texts.append(build_entity_text(entity))

            payloads.append({
                "phrase": entity["phrase"],
                "table_name": entity["table_name"],
                "source": entity["source"]
            })

            point_ids.append(
                stable_point_id_entity(
                    entity["phrase"],
                    entity["table_name"]
                )
            )

        if texts:

            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True
            ).tolist()

            if vectors and len(vectors[0]) != VECTOR_DIM:
                log.error(
                    f"Dimension mismatch: model produced {len(vectors[0])} "
                    f"but VECTOR_DIM={VECTOR_DIM}"
                )
                sys.exit(1)

            coll_name = "semantic_entities"

            ensure_collection(
                client,
                coll_name,
                recreate=args.recreate
            )

            for i in range(0, len(vectors), BATCH_SIZE):

                batch_points = [
                    qmodels.PointStruct(
                        id=point_ids[j],
                        vector=vectors[j],
                        payload=payloads[j],
                    )
                    for j in range(i, min(i + BATCH_SIZE, len(vectors)))
                ]

                client.upsert(
                    collection_name=coll_name,
                    points=batch_points
                )

                log.info(
                    f"  Upserted "
                    f"{min(i + BATCH_SIZE, len(vectors))}"
                    f"/{len(vectors)} to {coll_name}"
                )

    log.info("✅ Done.")


if __name__ == "__main__":
    main()