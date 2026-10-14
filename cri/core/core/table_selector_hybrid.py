from __future__ import annotations

import logging
import math
import re
import string
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

import os
log = logging.getLogger(__name__)
try:
    from core.meta_loader import get_module_prefixes
except Exception:
    get_module_prefixes = None

# SCHEMA_COLLECTION is shared across all modules — column-level vectors
# for every module all go into one collection.
SCHEMA_COLLECTION = os.getenv("QDRANT_SCHEMA_COLLECTION", "schema_columns")

# Per-module table metadata lives in separate Qdrant collections.
# Collection name = "<module_id_lower><suffix>"
# where suffix comes from MODULE_COLLECTION_SUFFIX env var (default: "_tables").
# e.g.  MFC → "mfc_tables"   ALM → "alm_tables"   USR → "usr_tables"
#
# TABLE_PREFIX env var drives which modules are known:
#   TABLE_PREFIX=MFC,ALM,USR
#
# You can also hardcode overrides per module via:
#   MFC_QDRANT_COLLECTION=mfc_tables
#   ALM_QDRANT_COLLECTION=alm_tables
# These take precedence over the auto-derived name.

_MODULE_COLLECTION_SUFFIX = os.getenv("MODULE_COLLECTION_SUFFIX", "_tables")

# After the MFC/PRD/SEQ config merge, embed_and_upload.py writes every
# module's table-level vectors into ONE combined collection derived from
# the combined narrow file's name (prod_narrow.json -> "prod_tables"),
# instead of one collection per TABLE_PREFIX code. If that combined
# collection name is set, every module code routes to it — the per-prefix
# derivation below (mfc_tables / prd_tables / seq_tables) is now stale and
# would query collections that no longer get written to.
_COMBINED_TABLES_COLLECTION = os.getenv("COMBINED_TABLES_COLLECTION", "prod_tables")


def _get_module_collection(module_id: str) -> str:
    """
    Return the Qdrant collection name for a given module ID.

    Resolution order:
      1. <MODULE_ID>_QDRANT_COLLECTION env var  (explicit override — still
         supported for anyone running a non-merged / per-module setup)
      2. COMBINED_TABLES_COLLECTION (default "prod_tables") if set — all
         modules route here after the MFC/PRD/SEQ merge.
      3. <module_id.lower()><MODULE_COLLECTION_SUFFIX>  (legacy per-module
         auto-derived name, e.g. MFC -> mfc_tables) — only reached if
         COMBINED_TABLES_COLLECTION is explicitly unset.
    """
    override_key = f"{module_id.upper()}_QDRANT_COLLECTION"
    override     = os.getenv(override_key, "").strip()
    if override:
        return override
    if _COMBINED_TABLES_COLLECTION:
        return _COMBINED_TABLES_COLLECTION
    return f"{module_id.lower()}{_MODULE_COLLECTION_SUFFIX}"


def _get_known_module_collections() -> dict:
    """
    Build {module_id: collection_name} for all active modules.
    Used for health-check logging at startup.
    """
    if get_module_prefixes is not None:
        codes = get_module_prefixes()
    else:
        raw = os.getenv("TABLE_PREFIX", "MFC").strip()
        codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    return {code: _get_module_collection(code) for code in codes}


# Fallback static data so resolve_tables_hybrid can return candidates even
# when the Qdrant / embedder is unavailable (no more empty-list → 400).
_MODULE_DATA: Optional[dict] = None


# ======================================================================
# BM25 Index
# ======================================================================
# Pure-Python BM25 (Okapi BM25) — no external dependency beyond stdlib.
# Built from each table's textual corpus:
#   table_name + description + use_case + query_patterns + column_names
#
# Scores are min-max normalised to [0, 1] before being mixed with the
# vector scores, so the bm25_weight parameter is directly comparable.
# ======================================================================

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "into", "through", "that",
    "this", "it", "its", "and", "or", "but", "not", "no", "nor",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop-words, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _STOP_WORDS]
    return tokens


class BM25Index:
    """
    Okapi BM25 index for a fixed set of (table_name, document) pairs.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation factor (typical: 1.2–2.0).
    b  : float
        Length normalisation factor (typical: 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._table_names: List[str] = []
        self._corpus_tokens: List[List[str]] = []
        self._df: Dict[str, int] = defaultdict(int)          # doc-freq per term
        self._doc_len: List[int] = []
        self._avg_doc_len: float = 0.0
        self._n_docs: int = 0
        self._built: bool = False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def build(self, table_corpus: Dict[str, str]) -> None:
        """
        Parameters
        ----------
        table_corpus : dict
            ``{table_name: combined_text}``
            Where ``combined_text`` should be a rich concatenation of all
            textual metadata fields for that table.
        """
        self._table_names = list(table_corpus.keys())
        self._corpus_tokens = []
        self._df = defaultdict(int)
        self._doc_len = []

        for table in self._table_names:
            tokens = _tokenize(table_corpus[table])
            self._corpus_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            for term in set(tokens):              # unique terms per doc
                self._df[term] += 1

        self._n_docs = len(self._table_names)
        self._avg_doc_len = (
            sum(self._doc_len) / self._n_docs if self._n_docs > 0 else 1.0
        )
        self._built = True
        log.debug(
            f"BM25Index built: {self._n_docs} docs, "
            f"avg_len={self._avg_doc_len:.1f}, vocab={len(self._df)}"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, query: str, candidates: List[str]) -> Dict[str, float]:
        """
        Returns BM25 scores (min-max normalised to [0, 1]) for every table
        in *candidates*.  Tables not in the index get score 0.
        """
        if not self._built:
            return {t: 0.0 for t in candidates}

        query_tokens = _tokenize(query)
        if not query_tokens:
            return {t: 0.0 for t in candidates}

        # Map table_name → index position for fast lookup
        name_to_idx = {n: i for i, n in enumerate(self._table_names)}

        raw: Dict[str, float] = {}
        for table in candidates:
            idx = name_to_idx.get(table)
            if idx is None:
                raw[table] = 0.0
                continue

            doc_tokens = self._corpus_tokens[idx]
            doc_len = self._doc_len[idx]
            tf_map: Dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1

            bm25_score = 0.0
            for term in query_tokens:
                if term not in self._df:
                    continue
                df = self._df[term]
                idf = math.log(
                    (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                tf = tf_map.get(term, 0)
                tf_norm = (
                    tf * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len))
                )
                bm25_score += idf * tf_norm

            raw[table] = bm25_score

        # Min-max normalise so scores land in [0, 1] — same range as cosine sim
        values = list(raw.values())
        min_v, max_v = min(values), max(values)
        span = max_v - min_v
        if span < 1e-9:
            return {t: 0.0 for t in candidates}
        return {t: (v - min_v) / span for t, v in raw.items()}


# ======================================================================
# Per-subgroup BM25 index cache
# Keyed by (module_name, subgroup_name) so each subgroup gets its own
# index built from its own candidate tables.
# ======================================================================
_bm25_cache: Dict[Tuple[str, str], BM25Index] = {}


def _build_corpus_text(table_name: str, payload: dict) -> str:
    """
    Concatenates all textual metadata fields from a Qdrant MFC payload into
    one string suitable for BM25 tokenisation.

    Fields used (all optional — missing fields are silently skipped):
      table_name, description, use_case, query_patterns, do_not_use_when,
      columns (list of dicts with 'name' / 'description' keys)
    """
    parts: List[str] = [table_name.replace("_", " ")]

    for field in ("description", "use_case"):
        val = payload.get(field)
        if isinstance(val, str) and val:
            parts.append(val)

    for field in ("query_patterns", "do_not_use_when"):
        val = payload.get(field)
        if isinstance(val, list):
            parts.extend(str(v) for v in val if v)

    columns = payload.get("columns", [])
    if isinstance(columns, list):
        for col in columns:
            if isinstance(col, dict):
                parts.append(col.get("name", ""))
                parts.append(col.get("description", ""))

    return " ".join(p for p in parts if p)


class HybridTableSelector:
    def __init__(
        self,
        module_data: dict,
        embedder: SentenceTransformer,
        qdrant_client: QdrantClient,
        meta_weight: float = 0.4,
        schema_weight: float = 0.1,
        bm25_weight: float = 0.3,
        top_k: int = 3,
        second_table_threshold: float = 0.65,
        # enrichment parameters
        do_not_use_penalty: float = 0.2,
        query_pattern_boost: float = 1.3,
        link_penalty: float = 0.5,
        relation_keywords: List[str] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        # ── Parent-child hierarchy expansion ──────────────────────────
        enable_hierarchy_expansion: bool = True,
        hierarchy_trigger_score: float = 0.55,
        max_hierarchy_depth: int = 4,
    ):
        # Validate weights — must sum to 1.0
        total = meta_weight + schema_weight + bm25_weight
        if abs(total - 1.0) > 1e-6:
            log.warning(
                f"Weights sum to {total:.3f} (not 1.0). "
                "Consider normalising meta+schema+bm25 = 1.0"
            )

        self.module_data = module_data
        self.embedder = embedder
        self.client = qdrant_client
        self.meta_weight = meta_weight
        self.schema_weight = schema_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.second_table_threshold = second_table_threshold
        self.do_not_use_penalty = do_not_use_penalty
        self.query_pattern_boost = query_pattern_boost
        self.link_penalty = link_penalty
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.relation_keywords = relation_keywords or [
            "with", "related", "join", "linked", "associated", "belongs",
            "against", "between", "via", "through", "using", "relationship"
        ]

        # ── Parent-child hierarchy expansion settings ──────────────────
        # Most tables in schemas like MFC are LINK/child tables hanging off a
        # MASTER root (e.g. MFC_L_OPERATION_TOOL -> MFC_L_MBOM_LA_OPERATION ->
        # MFC_L_MBOM -> MFC_M_MBOM_HEADER). Independent per-table scoring will
        # often rank the deepest, most semantically-specific child highest and
        # never surface the ancestor chain needed to actually join/identify it.
        # This expansion walks `parent_entity` (from the narrow JSON) for any
        # confidently-selected table and injects its ancestor chain into the
        # candidate set *before* top_k truncation happens downstream.
        self.enable_hierarchy_expansion = enable_hierarchy_expansion
        self.hierarchy_trigger_score = hierarchy_trigger_score
        self.max_hierarchy_depth = max_hierarchy_depth
        self._narrow_index: Dict[str, dict] = self._build_narrow_index(module_data)

        log.info(
            f"HybridTableSelector ready "
            f"(meta={meta_weight}, schema={schema_weight}, bm25={bm25_weight}, "
            f"top_k={top_k}, gap_threshold={second_table_threshold}, "
            f"do_not_use_penalty={do_not_use_penalty}, "
            f"query_pattern_boost={query_pattern_boost}, "
            f"link_penalty={link_penalty}, "
            f"bm25_k1={bm25_k1}, bm25_b={bm25_b}, "
            f"hierarchy_expansion={enable_hierarchy_expansion}, "
            f"hierarchy_trigger={hierarchy_trigger_score}, "
            f"max_hierarchy_depth={max_hierarchy_depth}, "
            f"narrow_index_size={len(self._narrow_index)})"
        )

    # ------------------------------------------------------------------
    # Narrow-JSON table index (table_name -> full narrow entry)
    # Built once from module_data so parent_entity / table_role /
    # parent_entity_status are available with zero Qdrant dependency —
    # mirrors chatbot_backend.py's NARROW_TABLE_INDEX so both layers of
    # the pipeline agree on the same hierarchy data.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_narrow_index(module_data: dict) -> Dict[str, dict]:
        index: Dict[str, dict] = {}
        for mod_val in (module_data or {}).get("modules", {}).values():
            for sg_val in mod_val.get("subgroups", {}).values():
                sg_list = sg_val.get("tables", sg_val) if isinstance(sg_val, dict) else sg_val
                if not isinstance(sg_list, list):
                    continue
                for tbl in sg_list:
                    if not isinstance(tbl, dict):
                        continue
                    tname = tbl.get("table_name", "")
                    if tname:
                        index[tname] = tbl
        return index

    # ------------------------------------------------------------------
    # Walk parent_entity chain up to the MASTER root (or max depth / a
    # missing / cyclical link, whichever comes first).
    # ------------------------------------------------------------------
    def _walk_parent_chain(self, table: str) -> List[str]:
        chain: List[str] = []
        visited = {table}
        current = table
        for _ in range(self.max_hierarchy_depth):
            entry = self._narrow_index.get(current)
            if not entry:
                break
            parent = entry.get("parent_entity")
            status = entry.get("parent_entity_status")
            if not parent or status != "present":
                if parent and status != "present":
                    log.debug(
                        f"[hierarchy] '{current}' declares parent '{parent}' but "
                        f"status='{status}' — chain stops here (parent not resolvable)"
                    )
                break
            if parent in visited:
                log.warning(f"[hierarchy] Cycle detected in parent_entity chain at '{parent}' — stopping")
                break
            chain.append(parent)
            visited.add(parent)
            current = parent
        return chain

    # ------------------------------------------------------------------
    # Candidate lookup
    # ------------------------------------------------------------------
    def _get_candidate_tables(self, module_name: str, subgroup_name: str) -> List[str]:
        modules = self.module_data.get("modules", {})
        module = modules.get(module_name, {})
        subgroups = module.get("subgroups", {})
        subgroup_value = subgroups.get(subgroup_name, [])
        
        # Handle both v1 (bare list of strings) and v2 (dict with scope_hard_block_keywords + tables as list of dicts)
        if isinstance(subgroup_value, dict):
            tables = subgroup_value.get("tables", [])
            # v2: tables is a list of dicts with 'table_name' key
            if tables and isinstance(tables[0], dict):
                return [t["table_name"] for t in tables]
            # v2 alternative: tables might be a list of strings
            return tables
        # v1: bare list of strings
        return subgroup_value

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        vec = self.embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ------------------------------------------------------------------
    # Meta retrieval – returns vector score AND full payload
    # ------------------------------------------------------------------
    def _meta_info(
        self,
        query_vec: List[float],
        candidates: List[str],
        module_id: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict: table_name -> {'score': float, 'payload': dict}

        module_id is used to resolve the correct per-module Qdrant collection.
        Falls back to the first module in TABLE_PREFIX if not provided.
        """
        if not candidates:
            return {}

        collection = _get_module_collection(module_id) if module_id else _get_module_collection(
            os.getenv("TABLE_PREFIX", "MFC").split(",")[0].strip()
        )
        log.debug(f"[meta_info] querying collection '{collection}' for module '{module_id}'")

        filter_cond = Filter(
            must=[
                FieldCondition(
                    key="table_name",
                    match=MatchAny(any=candidates)
                )
            ]
        )
        try:
            results = self.client.query_points(
                collection_name=collection,
                query=query_vec,
                query_filter=filter_cond,
                limit=len(candidates),
                with_payload=True,
            )
        except Exception as e:
            log.warning(f"[meta_info] Qdrant query failed for collection '{collection}': {e}")
            return {}

        points = results.points if hasattr(results, "points") else results
        table_info = {}
        for hit in points:
            table_name = hit.payload["table_name"]
            if table_name not in table_info or hit.score > table_info[table_name]["score"]:
                table_info[table_name] = {"score": hit.score, "payload": hit.payload}
        return table_info

    # ------------------------------------------------------------------
    # Schema scoring – blended max+mean
    # ------------------------------------------------------------------
    def _schema_similarities(self, query_vec: List[float], candidates: List[str]) -> Dict[str, float]:
        if not candidates:
            return {}
        filter_cond = Filter(
            must=[
                FieldCondition(
                    key="table_name",
                    match=MatchAny(any=candidates)
                )
            ]
        )
        results = self.client.query_points(
            collection_name=SCHEMA_COLLECTION,
            query=query_vec,
            query_filter=filter_cond,
            limit=len(candidates) * 10,
            with_payload=["table_name"],
        )
        points = results.points if hasattr(results, "points") else results

        table_scores: Dict[str, List[float]] = {t: [] for t in candidates}
        for hit in points:
            table = hit.payload["table_name"]
            if table in table_scores:
                table_scores[table].append(hit.score)

        blended: Dict[str, float] = {}
        for table, scores in table_scores.items():
            if not scores:
                blended[table] = 0.0
            else:
                max_score = max(scores)
                mean_score = sum(scores) / len(scores)
                blended[table] = 0.6 * max_score + 0.4 * mean_score
        return blended

    # ------------------------------------------------------------------
    # BM25 scoring                                    ❗ NEW
    # ------------------------------------------------------------------
    def _get_or_build_bm25(
        self,
        module_name: str,
        subgroup_name: str,
        meta_infos: Dict[str, Dict[str, Any]],
        candidates: List[str],
    ) -> BM25Index:
        """
        Returns a cached BM25Index for (module_name, subgroup_name).

        If the index doesn't exist yet (first query for this subgroup) it is
        built on-the-fly from the payloads already retrieved by _meta_info.

        The corpus text for each table is built from its module payload:
            table_name + description + use_case + query_patterns + columns
        """
        cache_key = (module_name, subgroup_name)
        if cache_key in _bm25_cache:
            return _bm25_cache[cache_key]

        # Build corpus from whatever payload we have; fall back to table_name only
        corpus: Dict[str, str] = {}
        for table in candidates:
            payload = meta_infos.get(table, {}).get("payload", {})
            corpus[table] = _build_corpus_text(table, payload)

        idx = BM25Index(k1=self.bm25_k1, b=self.bm25_b)
        idx.build(corpus)
        _bm25_cache[cache_key] = idx
        log.info(f"BM25 index built and cached for ({module_name}, {subgroup_name})")
        return idx

    def _bm25_scores(
        self,
        question: str,
        module_name: str,
        subgroup_name: str,
        meta_infos: Dict[str, Dict[str, Any]],
        candidates: List[str],
    ) -> Dict[str, float]:
        """Returns normalised BM25 scores for each candidate table."""
        idx = self._get_or_build_bm25(module_name, subgroup_name, meta_infos, candidates)
        return idx.score(question, candidates)

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalise a score dict to [0, 1]."""
        if not scores:
            return scores
        min_v = min(scores.values())
        max_v = max(scores.values())
        span = max_v - min_v
        if span < 1e-9:
            return {k: 0.0 for k in scores}
        return {k: (v - min_v) / span for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Module-code resolution
    # ------------------------------------------------------------------
    def _resolve_module_code(self, module_name: str) -> str:
        """
        Map a narrow-JSON module key (e.g. "access_control", "user_profiles")
        to its TABLE_PREFIX code (e.g. "USR", "MFC") so _meta_info can find
        the correct Qdrant collection.

        Resolution order:
          1. <MODULE_NAME_UPPER>_MODULE_CODE env var explicit override.
             e.g. ACCESS_CONTROL_MODULE_CODE=USR
          2. Walk MODULE_DATA: each narrow file's modules carry a `module_prefix`
             field injected by meta_loader. Return that when module_name matches.
          3. Check known TABLE_PREFIX codes directly (handles plain "MFC" etc.).
          4. Search all modules for module_name as a subgroup key.
          5. Fall back to module_name.upper().
        """
        # ── 1. Explicit env var override ──────────────────────────────────────
        env_key = f"{module_name.upper().replace('-', '_')}_MODULE_CODE"
        override = os.getenv(env_key, "").strip()
        if override:
            log.debug(f"[_resolve_module_code] '{module_name}' → '{override}' (env {env_key})")
            return override.upper()

        # ── 2. Walk MODULE_DATA for a `module_prefix` annotation ─────────────
        modules = self.module_data.get("modules", {})
        entry = modules.get(module_name)
        if entry and entry.get("module_prefix"):
            return entry["module_prefix"].upper()

        # ── 3. Direct match against known TABLE_PREFIX codes ─────────────────
        known_codes = set(get_module_prefixes()) if get_module_prefixes is not None else {
            c.strip().upper()
            for c in os.getenv("TABLE_PREFIX", "MFC").split(",")
            if c.strip()
        }
        if module_name.upper() in known_codes:
            return module_name.upper()

        # ── 4. Search all modules: if module_name is a subgroup key somewhere,
        #       use that parent module's prefix.
        for mod_key, mod_val in modules.items():
            if module_name in mod_val.get("subgroups", {}):
                prefix = mod_val.get("module_prefix", "")
                if prefix:
                    log.debug(
                        f"[_resolve_module_code] '{module_name}' resolved as subgroup of "
                        f"'{mod_key}' → prefix '{prefix}'"
                    )
                    return prefix.upper()

        # ── 5. Last resort ────────────────────────────────────────────────────
        log.warning(
            f"[_resolve_module_code] Could not resolve TABLE_PREFIX code for "
            f"module_name='{module_name}'. Falling back to '{module_name.upper()}'. "
            f"Set {env_key}=<CODE> in .env to fix this explicitly."
        )
        return module_name.upper()

    # ------------------------------------------------------------------
    # Main selection logic
    # ------------------------------------------------------------------
    def select_tables(
        self,
        question: str,
        module_name: str,
        subgroup_name: str,
        return_scores: bool = False,
    ) -> "List[str] | Tuple[List[str], Dict[str, float]]": 
        candidates = self._get_candidate_tables(module_name, subgroup_name)
        if not candidates:
            return ([], {}) if return_scores else []
        if len(candidates) == 1:
            log.info(f"Only 1 table available, returning: {candidates}")
            return (candidates, {candidates[0]: 1.0}) if return_scores else candidates

        query_vec = self._embed(question)

        # 1️⃣ Vector similarity + full metadata — routed to per-module collection.
        # module_name is the narrow JSON module key (e.g. "access_control",
        # "user_profiles", "engineering_bom").  We need the TABLE_PREFIX code
        # (e.g. "MFC", "USR") to derive the correct Qdrant collection name.
        #
        # Resolution order:
        #   a) Walk MODULE_DATA to find which top-level narrow file contains
        #      this module key, then map it to its TABLE_PREFIX code via the
        #      module's own `module_prefix` field (set by meta_loader).
        #   b) Check if module_name itself is already a known TABLE_PREFIX code.
        #   c) Last resort: use module_name as-is (original behaviour).
        _module_code = self._resolve_module_code(module_name)
        meta_infos = self._meta_info(query_vec, candidates, module_id=_module_code)

        # 2️⃣ Schema column-level vector scores
        schema_scores = self._schema_similarities(query_vec, candidates)

        # 3️⃣ BM25 keyword scores (already normalised to [0,1] by BM25Index.score())
        bm25_scores = self._bm25_scores(
            question, module_name, subgroup_name, meta_infos, candidates
        )
        log.debug(f"BM25 scores: { {t: round(s, 4) for t, s in bm25_scores.items()} }")

        # 4️⃣ Normalise meta + schema to [0,1] so all three signals are on
        #    the same scale before being combined with weights.
        raw_meta_scores = {
            t: (meta_infos[t]["score"] if t in meta_infos else 0.0)
            for t in candidates
        }
        norm_meta   = self._normalize(raw_meta_scores)
        norm_schema = self._normalize(schema_scores)

        log.debug(f"Norm meta:   { {t: round(s, 4) for t, s in norm_meta.items()} }")
        log.debug(f"Norm schema: { {t: round(s, 4) for t, s in norm_schema.items()} }")

        # 5️⃣ Combined score = meta + schema + bm25
        raw_combined = {}
        for table in candidates:
            meta_score   = norm_meta.get(table, 0.0)
            schema_score = norm_schema.get(table, 0.0)
            bm25_score   = bm25_scores.get(table, 0.0)
            raw_combined[table] = (
                self.meta_weight   * meta_score
                + self.schema_weight * schema_score
                + self.bm25_weight   * bm25_score
            )

        # 6️⃣ Apply do_not_use_when penalty
        for table in candidates:
            payload = meta_infos.get(table, {}).get("payload", {})
            do_not_use = payload.get("do_not_use_when", [])
            if do_not_use:
                for pattern in do_not_use:
                    if pattern.lower() in question.lower():
                        raw_combined[table] *= self.do_not_use_penalty
                        log.debug(f"Negative match on '{table}': pattern '{pattern}' → penalty applied")
                        break

        # 7️⃣ Apply query_pattern boost
        for table in candidates:
            payload = meta_infos.get(table, {}).get("payload", {})
            patterns = payload.get("query_patterns", [])
            if patterns:
                for pat in patterns:
                    if pat.lower() in question.lower():
                        raw_combined[table] *= self.query_pattern_boost
                        log.debug(f"Query pattern match on '{table}': '{pat}' → boost applied")
                        break

        # 8️⃣ Apply LINK table penalty for simple queries
        is_relationship_query = any(
            kw in question.lower() for kw in self.relation_keywords
        )
        for table in candidates:
            payload = meta_infos.get(table, {}).get("payload", {})
            if payload.get("table_type") == "LINK" and not is_relationship_query:
                raw_combined[table] *= self.link_penalty
                log.debug(f"LINK table '{table}' penalised (simple query)")

        # Sort by adjusted scores
        sorted_tables = sorted(raw_combined.items(), key=lambda x: x[1], reverse=True)

        # ✅ FIXED: Always return top K tables — no gap filter, no ratio threshold.
        # The old second_table_threshold caused multi-table queries to collapse to
        # 1 table even when top_k=5, because any score gap > threshold silently
        # dropped the second table.  Entity extraction + graph resolution downstream
        # handle join decisions — the selector's job is just to shortlist candidates.
        selected = [t for t, _ in sorted_tables[:self.top_k]]

        # Safety: ensure at least 2 tables when the pool has them,
        # so the entity→graph pipeline always has candidates to work with.
        if len(selected) < 2 and len(sorted_tables) >= 2:
            selected = [t for t, _ in sorted_tables[:2]]

        # 9️⃣ Parent-child hierarchy expansion — walk `parent_entity` for any
        # confidently-selected table and inject its ancestor chain into the
        # returned set *before* it gets handed off downstream. Without this,
        # a deep LINK/child table (e.g. MFC_L_OPERATION_TOOL) can win top_k on
        # semantic score alone while its MASTER root (e.g. MFC_M_MBOM_HEADER)
        # never appears anywhere in the candidate pool, and no amount of
        # downstream FK-boosting can recover a table that was never selected.
        if self.enable_hierarchy_expansion:
            chain_additions: List[str] = []
            for table in list(selected):
                score = raw_combined.get(table, 0.0)
                if score < self.hierarchy_trigger_score:
                    continue
                chain = self._walk_parent_chain(table)
                for ancestor in chain:
                    if ancestor not in selected and ancestor not in chain_additions:
                        chain_additions.append(ancestor)
                        # Ancestors are structurally required, not just
                        # semantically similar — give them a score high
                        # enough to survive downstream threshold / softmax /
                        # dominance filters instead of a raw 0.0.
                        raw_combined[ancestor] = max(raw_combined.get(ancestor, 0.0), score * 0.9)
            if chain_additions:
                log.info(
                    f"[hierarchy] Expanded selection with ancestor chain(s): {chain_additions} "
                    f"(triggered by tables scoring >= {self.hierarchy_trigger_score})"
                )
                selected = selected + chain_additions
                # Keep sorted_tables in sync so return_scores reflects the
                # injected ancestor scores too.
                sorted_tables = sorted(raw_combined.items(), key=lambda x: x[1], reverse=True)

        log.info(
            f"Selected tables for '{question}': {selected} "
            f"(all scores: {[(t, round(s, 4)) for t, s in sorted_tables]})"
        )

        if return_scores:
            score_dict = {t: raw_combined.get(t, s) for t, s in sorted_tables if t in selected}
            # Ensure every selected table has an entry even if it wasn't in
            # sorted_tables' original candidate pool (hierarchy ancestors
            # pulled in from outside this subgroup's candidate list).
            for t in selected:
                score_dict.setdefault(t, raw_combined.get(t, 0.0))
            return selected, score_dict
        return selected


# ------------------------------------------------------------------
# Singleton management (unchanged public interface)
# ------------------------------------------------------------------
_selector: Optional[HybridTableSelector] = None


def init_table_selector(
    module_data: dict,
    embedder: Optional[SentenceTransformer] = None,
    qdrant_client: Optional[QdrantClient] = None,
    meta_weight: float = 0.4,
    schema_weight: float = 0.1,
    bm25_weight: float = 0.3,
    top_k: int = 3,
    second_table_threshold: float = 0.65,
    do_not_use_penalty: float = 0.2,
    query_pattern_boost: float = 1.3,
    link_penalty: float = 0.5,
    relation_keywords: List[str] = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    enable_hierarchy_expansion: bool = True,
    hierarchy_trigger_score: float = 0.55,
    max_hierarchy_depth: int = 4,
) -> None:
    global _selector, _MODULE_DATA

    _MODULE_DATA = module_data

    if embedder is None or qdrant_client is None:
        log.warning("Hybrid selector not fully initialised – using fallback (returns all candidates)")
        _selector = None
        return

    # Health check: verify all per-module collections + schema collection
    try:
        module_collections = _get_known_module_collections()
        for mod_id, coll_name in module_collections.items():
            try:
                count = qdrant_client.count(collection_name=coll_name).count
                log.info(f"Qdrant [{mod_id}] collection '{coll_name}': {count} pts")
                if count == 0:
                    log.warning(
                        f"⚠️  '{coll_name}' is empty → meta scoring for {mod_id} "
                        f"will always be zero. Run the {mod_id} indexer first."
                    )
            except Exception as e:
                log.warning(
                    f"⚠️  Could not verify collection '{coll_name}' for module {mod_id}: {e}"
                )
        schema_count = qdrant_client.count(collection_name=SCHEMA_COLLECTION).count
        log.info(f"Qdrant schema collection '{SCHEMA_COLLECTION}': {schema_count} pts")
        if schema_count == 0:
            log.warning(f"⚠️  '{SCHEMA_COLLECTION}' is empty → schema scoring will always be zero")
    except Exception as e:
        log.warning(f"Could not verify Qdrant collections: {e}")

    _selector = HybridTableSelector(
        module_data=module_data,
        embedder=embedder,
        qdrant_client=qdrant_client,
        meta_weight=meta_weight,
        schema_weight=schema_weight,
        bm25_weight=bm25_weight,
        top_k=top_k,
        second_table_threshold=second_table_threshold,
        do_not_use_penalty=do_not_use_penalty,
        query_pattern_boost=query_pattern_boost,
        link_penalty=link_penalty,
        relation_keywords=relation_keywords,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        enable_hierarchy_expansion=enable_hierarchy_expansion,
        hierarchy_trigger_score=hierarchy_trigger_score,
        max_hierarchy_depth=max_hierarchy_depth,
    )

    log.info("BM25 indexes will be built lazily on first query per subgroup.")


def resolve_tables_hybrid(
    question: str,
    module_name: str,
    subgroup_name: str,
    return_scores: bool = False,
) -> "List[str] | Tuple[List[str], Dict[str, float]]": 
    """
    Returns the best-matching tables for the given question and subgroup.
    When return_scores=True returns (tables, score_dict) with the actual
    combined hybrid scores so callers can apply softmax + probability filters.
    Falls back to returning all candidate tables when Qdrant is unavailable.
    """
    if _selector is None:
        if _MODULE_DATA is None:
            log.warning("No module_data for fallback – returning empty list")
            return ([], {}) if return_scores else []
        modules  = _MODULE_DATA.get("modules", {})
        module   = modules.get(module_name, {})
        subgroups = module.get("subgroups", {})
        tables   = subgroups.get(subgroup_name, [])
        candidates = [t["table_name"] for t in tables]
        log.warning(
            f"Hybrid selector unavailable – fallback to all {len(candidates)} tables: {candidates}"
        )
        if return_scores:
            # Assign equal scores in fallback — softmax will flatten them anyway
            scores = {t: 1.0 for t in candidates}
            return candidates, scores
        return candidates

    return _selector.select_tables(question, module_name, subgroup_name, return_scores=return_scores)


# ------------------------------------------------------------------
# Cache invalidation helper (call if module_data changes at runtime)
# ------------------------------------------------------------------
def clear_bm25_cache() -> None:
    """Clears the per-subgroup BM25 index cache (e.g. after a data reload)."""
    global _bm25_cache
    _bm25_cache.clear()
    log.info("BM25 index cache cleared.")