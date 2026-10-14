from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required. Install with: pip install faiss-cpu"
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. "
        "Install with: pip install sentence-transformers"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise ImportError("tqdm is required. Install with: pip install tqdm") from exc


# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_column_embeddings")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "mfc_narrow.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "embeddings"
CATEGORY_SOURCE_PRIORITY: List[Tuple[str, str]] = [
    ("foreign_keys", "Foreign Key"),
    ("time_columns", "Date"),
    ("status_columns", "Status"),
    ("quantity_columns", "Quantity"),
    ("value_columns", "Value"),
]


NAME_CATEGORY_HINTS: List[Tuple[str, str]] = [
    ("_id", "Identifier"),
    ("id", "Identifier"),
    ("is_", "Boolean"),
    ("has_", "Boolean"),
    ("flag", "Boolean"),
    ("description", "Description"),
    ("desc", "Description"),
    ("name", "Description"),
    ("reason", "Description"),
    ("code", "Identifier"),
    ("number", "Identifier"),
    ("no_", "Identifier"),
]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class ColumnRecord:
    """Represents a single column embedding unit prior to vectorization."""

    module: str
    table: str
    column: str
    category: str
    embedding_text: str


@dataclass
class ModuleStats:
    """Tracks per-module processing statistics for the summary report."""

    module: str
    tables_processed: int = 0
    columns_embedded: int = 0
    output_path: str = ""


# --------------------------------------------------------------------------- #
# Category inference
# --------------------------------------------------------------------------- #

class ColumnCategoryInferer:
    """Infers a column's semantic category from table metadata."""

    @staticmethod
    def _normalize_key_list(raw: Any) -> List[str]:
        """Normalizes a table's category-source field into a flat list of
        column-name strings, regardless of whether it was authored as a
        list of strings or a list/dict of foreign-key objects."""
        names: List[str] = []
        if raw is None:
            return names
        if isinstance(raw, dict):
            names.extend(str(k) for k in raw.keys())
            return names
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    
                    candidate = item.get("column") or item.get("column_name")
                    if candidate:
                        names.append(str(candidate))
            return names
        return names

    @classmethod
    def infer(cls, column_name: str, table: Dict[str, Any]) -> str:
        """Infers the category for a given column using the documented
        priority order: foreign_keys > time_columns > status_columns >
        quantity_columns > value_columns > name-based heuristic."""
        for source_key, category_label in CATEGORY_SOURCE_PRIORITY:
            candidates = cls._normalize_key_list(table.get(source_key))
            if column_name in candidates:
                return category_label

        lowered = column_name.lower()
        for hint, category_label in NAME_CATEGORY_HINTS:
            if hint in lowered:
                return category_label

        return "General"


# --------------------------------------------------------------------------- #
# Embedding text construction
# --------------------------------------------------------------------------- #

class EmbeddingTextBuilder:
    """Builds the fixed-format embedding text for a single column.

    Field order matters here. SentenceTransformer mean-pools over every
    token in the sequence, and business_purpose / row_represents / use_when
    are IDENTICAL for every column belonging to the same table. If that
    boilerplate is long relative to the column-specific text (name +
    description), it dominates the pooled vector and every column in the
    table ends up embedding almost identically — killing FAISS's ability
    to discriminate between columns. It also risks being pushed past the
    model's 256-token limit for verbose tables, truncating the
    column-specific part off entirely.

    Fix: front-load the column-specific fields (highest discriminative
    value, guaranteed to survive truncation) and cap the shared table-level
    context so it can't drown them out.
    """

    _MAX_CONTEXT_CHARS = 220   # cap for business_purpose / row_represents
    _MAX_USE_WHEN_ITEMS = 4    # cap on how many use_when bullets are included

    @staticmethod
    def _clip(text: str, max_chars: int = 220) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def build(
        module_label: str,
        table_name: str,
        business_purpose: str,
        row_represents: str,
        column_name: str,
        column_description: str,
        category: str,
        use_when: List[str],
    ) -> str:
        clip = EmbeddingTextBuilder._clip
        capped_use_when = (use_when or [])[: EmbeddingTextBuilder._MAX_USE_WHEN_ITEMS]
        use_when_text = "\n".join(f"- {item}" for item in capped_use_when) if capped_use_when else "N/A"
        return (
            f"Column:\n{column_name}\n\n"
            f"Description:\n{column_description}\n\n"
            f"Category:\n{category}\n\n"
            f"Table:\n{table_name}\n\n"
            f"Module:\n{module_label}\n\n"
            f"Business Purpose:\n{clip(business_purpose)}\n\n"
            f"Row Represents:\n{clip(row_represents)}\n\n"
            f"Use When:\n{use_when_text}"
        )


# --------------------------------------------------------------------------- #
# Schema parsing
# --------------------------------------------------------------------------- #

class SchemaParser:
    """Parses the mfc_narrow.json schema into a flat list of ColumnRecords,
    grouped implicitly by module via the `module` field on each record."""

    def __init__(self, schema_path: Path):
        self.schema_path = schema_path

    def load(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        try:
            with self.schema_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {self.schema_path}: {exc}") from exc

    def parse(self) -> Dict[str, List[ColumnRecord]]:
        """Returns a dict mapping module_name -> list of ColumnRecords."""
        data = self.load()
        modules = data.get("modules", {})
        if not modules:
            raise ValueError("No 'modules' key found in schema file.")

        module_records: Dict[str, List[ColumnRecord]] = {}

        for module_name, module_obj in modules.items():
            module_label = module_obj.get("module_label", module_name)
            subgroups = module_obj.get("subgroups", {})
            records: List[ColumnRecord] = []

            for subgroup_name, subgroup_obj in subgroups.items():
                tables = subgroup_obj.get("tables", [])
                for table in tables:
                    records.extend(
                        self._build_records_for_table(module_name, module_label, table)
                    )

            module_records[module_name] = records
            logger.info(
                "Parsed module '%s': %d columns across %d subgroup(s).",
                module_name,
                len(records),
                len(subgroups),
            )

        return module_records

    @staticmethod
    def _build_records_for_table(
        module_name: str, module_label: str, table: Dict[str, Any]
    ) -> List[ColumnRecord]:
        table_name = table.get("table_name", "UNKNOWN_TABLE")
        business_purpose = table.get("business_purpose", "") or ""
        row_represents = table.get("row_represents", "") or ""
        use_when = table.get("use_when", []) or []
        columns = table.get("columns", {}) or {}

        records: List[ColumnRecord] = []
        for column_name, column_description in columns.items():
            category = ColumnCategoryInferer.infer(column_name, table)
            embedding_text = EmbeddingTextBuilder.build(
                module_label=module_label,
                table_name=table_name,
                business_purpose=business_purpose,
                row_represents=row_represents,
                column_name=column_name,
                column_description=str(column_description),
                category=category,
                use_when=use_when,
            )
            records.append(
                ColumnRecord(
                    module=module_name,
                    table=table_name,
                    column=column_name,
                    category=category,
                    embedding_text=embedding_text,
                )
            )
        return records


# --------------------------------------------------------------------------- #
# Embedding generation
# --------------------------------------------------------------------------- #

class ColumnEmbedder:
    """Wraps a SentenceTransformer model to batch-embed column texts."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading embedding model: %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc

        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                "Model dimension (%d) does not match expected dimension (%d).",
                actual_dim,
                EMBEDDING_DIM,
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generates L2-normalized embeddings for a list of texts, in batches."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required for cosine similarity via IndexFlatIP
        )
        return vectors.astype("float32")


# --------------------------------------------------------------------------- #
# FAISS index + metadata persistence
# --------------------------------------------------------------------------- #

class ModuleIndexWriter:
    """Builds and persists a FAISS IndexFlatIP index and metadata.json for
    a single module's column records."""

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def write(
        self,
        module_name: str,
        records: List[ColumnRecord],
        vectors: np.ndarray,
    ) -> str:
        module_dir = self.output_root / module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(vectors.shape[1] if vectors.size else EMBEDDING_DIM)
        if vectors.size:
            index.add(vectors)

        faiss_path = module_dir / "index.faiss"
        faiss.write_index(index, str(faiss_path))

        metadata = [
            {
                "id": idx,
                "module": rec.module,
                "table": rec.table,
                "column": rec.column,
                "category": rec.category,
                "embedding_text": rec.embedding_text,
            }
            for idx, rec in enumerate(records)
        ]
        metadata_path = module_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(module_dir)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class ColumnEmbeddingIndexBuilder:
    """Top-level orchestrator: parses the schema, generates embeddings, and
    writes module-wise FAISS indexes + metadata files."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = 64,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.stats: List[ModuleStats] = []

    def run(self) -> None:
        logger.info("Starting column embedding index build.")
        logger.info("Input schema: %s", self.input_path)
        logger.info("Output directory: %s", self.output_dir)

        parser = SchemaParser(self.input_path)
        module_records = parser.parse()

        total_columns = sum(len(recs) for recs in module_records.values())
        if total_columns == 0:
            logger.warning("No columns found in schema. Nothing to embed.")
            return

        embedder = ColumnEmbedder(model_name=self.model_name, batch_size=self.batch_size)
        writer = ModuleIndexWriter(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=total_columns, desc="Embedding columns", unit="col") as pbar:
            for module_name, records in module_records.items():
                if not records:
                    logger.info("Module '%s' has no columns; skipping.", module_name)
                    continue

                try:
                    texts = [rec.embedding_text for rec in records]
                    vectors = self._embed_in_batches(embedder, texts, pbar)
                    output_path = writer.write(module_name, records, vectors)

                    tables_processed = len({rec.table for rec in records})
                    self.stats.append(
                        ModuleStats(
                            module=module_name,
                            tables_processed=tables_processed,
                            columns_embedded=len(records),
                            output_path=output_path,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to build index for module '%s': %s", module_name, exc
                    )
                    continue

        self._print_summary()

    @staticmethod
    def _embed_in_batches(
        embedder: ColumnEmbedder, texts: List[str], pbar: "tqdm"
    ) -> np.ndarray:
        """Embeds texts in batches, updating the shared progress bar per batch."""
        batch_size = embedder.batch_size
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = embedder.embed(batch)
            all_vectors.append(vectors)
            pbar.update(len(batch))

        if not all_vectors:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")
        return np.vstack(all_vectors)

    def _print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("COLUMN EMBEDDING INDEX BUILD SUMMARY")
        print("=" * 70)
        total_tables = 0
        total_columns = 0
        for stat in self.stats:
            print(f"Module          : {stat.module}")
            print(f"  Tables processed : {stat.tables_processed}")
            print(f"  Columns embedded : {stat.columns_embedded}")
            print(f"  Output path      : {stat.output_path}")
            print("-" * 70)
            total_tables += stat.tables_processed
            total_columns += stat.columns_embedded
        print(f"TOTAL modules   : {len(self.stats)}")
        print(f"TOTAL tables    : {total_tables}")
        print(f"TOTAL columns   : {total_columns}")
        print("=" * 70)
        logger.info(
            "Build complete. %d modules, %d tables, %d columns embedded.",
            len(self.stats),
            total_tables,
            total_columns,
        )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a module-wise FAISS column embedding index from the "
            "mfc_narrow.json schema. Offline index build ONLY: no retrieval, "
            "SQL generation, join logic, or pipeline integration is performed."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Path to the schema JSON file (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Root output directory for module-wise FAISS indexes (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        builder = ColumnEmbeddingIndexBuilder(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            model_name=args.model,
            batch_size=args.batch_size,
        )
        builder.run()
        return 0
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Schema error: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during index build: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())