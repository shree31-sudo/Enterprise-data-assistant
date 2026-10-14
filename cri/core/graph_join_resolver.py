from __future__ import annotations

import json
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("GraphJoinResolver")

# ── env ───────────────────────────────────────────────────────────────────────
_NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "cncdb0804")

# Tunable thresholds (can be set in .env)
_JOIN_SCORE_THRESHOLD: float = float(os.getenv("JOIN_SCORE_THRESHOLD", "1.5"))
_JOIN_EMB_WEIGHT:      float = float(os.getenv("JOIN_EMB_WEIGHT",      "0.5"))


# ─────────────────────────────────────────────────────────────────────────────
# RELATIONAL KEYWORD PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
_RELATIONAL_PHRASES = re.compile(
    r"""\b(
        with\s+(?:its|their|the)
        | along\s+with
        | and\s+(?:its|their)
        | belonging\s+to
        | associated\s+with
        | linked\s+to
        | related\s+to
        | for\s+each\b
        | together\s+with
        | joined\s+(?:with|to|by)
        | cross[\s-]reference
        | across\s+\w+
        | between\s+\w+\s+and\s+\w+
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Pre-computed embedding key for the "multi-table relational question" concept
_RELATIONAL_ANCHOR_PHRASES = [
    "show items with their work orders",
    "list components and their associated operations",
    "find order details along with routing information",
    "get work order sequence and item assignment",
    "show BOM versions with their component details",
    "customer order history with product and shipping info",
]


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED CYPHER QUERY
# ─────────────────────────────────────────────────────────────────────────────
_CYPHER_WEIGHTED_PATH = """
MATCH p = (a:Table {name: $base})-[:JOINS_TO*1..4]-(b:Table {name: $target})
WITH p,
     reduce(score = 0, r IN relationships(p) |
         score + r.cost
     ) AS cost
RETURN p, cost
ORDER BY cost ASC
LIMIT 1
"""

_CYPHER_PATH_EXISTS = """
MATCH p = (a:Table {name: $base})-[:JOINS_TO*1..4]-(b:Table {name: $target})
RETURN count(p) > 0 AS exists
LIMIT 1
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — extract ordered join steps from a Neo4j path record
# ─────────────────────────────────────────────────────────────────────────────
def _path_to_steps(path_record) -> List[dict]:
    """
    Convert a neo4j Path object into the same step-dict format used by the
    existing build_join_clause logic:
        [{"from_table": ..., "to_table": ...,
          "from_column": ..., "to_column": ...}, ...]
    """
    steps = []
    rels  = list(path_record.relationships)
    nodes = list(path_record.nodes)
    for i, rel in enumerate(rels):
        start_name = nodes[i]["name"]
        end_name   = nodes[i + 1]["name"]
        # relationship direction may be reversed in undirected traversal
        if rel.start_node["name"] == start_name:
            fc, tc = rel["from_col"], rel["to_col"]
            ft, tt = start_name, end_name
        else:
            # traversed backwards — swap cols so the SQL JOIN is still correct
            fc, tc = rel["to_col"], rel["from_col"]
            ft, tt = start_name, end_name
        steps.append({
            "from_table":   ft,
            "to_table":     tt,
            "from_column":  fc,
            "to_column":    tc,
        })
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# JSON BFS FALLBACK  (identical logic to original chatbot_backend.py)
# ─────────────────────────────────────────────────────────────────────────────
class _JsonBFS:
    def __init__(self, join_paths: Dict, graph: Dict):
        self._jp    = join_paths
        self._graph = graph

    def direct(self, base: str, target: str) -> Optional[list]:
        return self._jp.get(base, {}).get(target)

    def transitive(self, base: str, target: str, depth: int = 3) -> Optional[List[list]]:
        q: deque = deque([(base, [])])
        visited  = {base.lower()}
        while q:
            cur, acc = q.popleft()
            if len(acc) >= depth:
                continue
            for tt, steps in self._jp.get(cur, {}).items():
                new_acc = acc + [steps]
                if tt.lower() == target.lower():
                    return new_acc
                if tt.lower() not in visited:
                    visited.add(tt.lower())
                    q.append((tt, new_acc))
        return None

    def find_path(self, base: str, target: str) -> Optional[List[dict]]:
        path = self.direct(base, target)
        if path:
            return path
        multi = self.transitive(base, target)
        if multi:
            return [step for seg in multi for step in seg]
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
def load_all_join_graphs() -> Dict:
    """
    Load and merge join_paths + graph sections from ALL module join graph files
    found in join_graphs/, filtered by TABLE_PREFIX in .env.

    Returns a single merged dict:
        { "join_paths": { ... all modules ... }, "graph": { ... all modules ... } }

    Use this when constructing GraphJoinResolver so the BFS fallback covers
    every module, not just MFC.
    """
    import json
    from pathlib import Path

    _PROJECT    = Path(__file__).resolve().parent.parent
    _JOIN_GRAPHS = _PROJECT / "data" / "join_graphs"  # consistent with graph_loader.py

    # build_join_graph.py now does ONE global fetch across every prefix in
    # TABLE_PREFIX (MFC, PRD, SEQ, ...) and writes a SINGLE combined file —
    # prod_join_graph.json — so that multi-hop paths can cross module
    # boundaries (e.g. PRD_L_REWORK_ORDER -> SEQ_L_WORK_ORDER). Load that
    # single file directly instead of looping per-prefix; per-prefix files
    # (mfc_join_graph.json / prd_join_graph.json / seq_join_graph.json) are
    # no longer written and would silently no-op here.
    fname = _JOIN_GRAPHS / "prod_join_graph.json"
    merged_jp: Dict = {}
    merged_g:  Dict = {}

    if not fname.exists():
        log.warning(
            "[GraphJoinResolver] %s not found — run build_join_graph.py "
            "first. Falling back to empty join graph.",
            fname.name,
        )
        return {"join_paths": merged_jp, "graph": merged_g}

    try:
        data = json.loads(fname.read_text(encoding="utf-8"))
        merged_jp = data.get("join_paths", {})
        merged_g  = data.get("graph", {})
        log.info("[GraphJoinResolver] Loaded combined join graph: %s", fname.name)
    except Exception as exc:
        log.error("[GraphJoinResolver] Failed to load %s: %s", fname.name, exc)
        return {"join_paths": {}, "graph": {}}

    log.info(
        "[GraphJoinResolver] Loaded join graph (module=PROD): "
        "%d base tables in join_paths, %d in graph.",
        len(merged_jp), len(merged_g),
    )
    return {"join_paths": merged_jp, "graph": merged_g}


class GraphJoinResolver:
    """
    Production join resolver with:
      1. Neo4j weighted path search (with JSON-BFS fallback)
      2. 3-gate classification: single / semantic / Neo4j
      3. Join-necessity check  (force simple if one table has all cols)
      4. Intent-aware path cache keyed by (base, target, intent)

    Pass join_graph=load_all_join_graphs() at construction so the BFS fallback
    covers all modules listed in TABLE_PREFIX (.env), not just MFC.
    """

    def __init__(
        self,
        meta:       Dict,          # merged META dict from meta_loader.load_all_meta()
        join_graph: Dict,          # merged dict from load_all_join_graphs()
        embedder=None,             # SentenceTransformer — optional, for Gate 2
    ):
        self._meta       = meta
        self._bfs        = _JsonBFS(
            join_graph.get("join_paths", {}),
            join_graph.get("graph",      {}),
        )
        self._embedder   = embedder
        self._driver     = None
        self._neo4j_ok   = False

        # path cache:  (base_table, target_table, intent) → List[dict] | None
        self._path_cache: Dict[Tuple, Optional[List[dict]]] = {}

        # pre-compute column sets per table for join-necessity check
        # meta is the flat table_name → table_info dict (already resolved by
        # _build_meta_from_module_data in chatbot_backend.py before being passed here)
        self._table_cols: Dict[str, set] = {
            tbl: set(tinfo.get("columns", {}).keys())
            for tbl, tinfo in meta.items()
            if isinstance(tinfo, dict) and "table_name" in tinfo
        }

        # optional: anchor embedding for semantic gate
        self._anchor_vecs: Optional[np.ndarray] = None
        if embedder is not None:
            self._build_anchor_vecs()

    # ── startup ───────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Try to connect to Neo4j. Returns True on success, False on failure.
        If False the resolver falls back to JSON-BFS transparently.
        """
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(
                _NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS)
            )
            self._driver.verify_connectivity()
            self._neo4j_ok = True
            log.info(f"[GraphJoinResolver] Neo4j connected at {_NEO4J_URI}")
        except ImportError:
            log.warning(
                "[GraphJoinResolver] neo4j package not installed "
                "(pip install neo4j). Using JSON-BFS fallback."
            )
        except Exception as exc:
            log.warning(
                f"[GraphJoinResolver] Neo4j unavailable ({exc}). "
                "Using JSON-BFS fallback — chatbot will still work."
            )
        return self._neo4j_ok

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def _build_anchor_vecs(self) -> None:
        try:
            vecs = self._embedder.encode(
                _RELATIONAL_ANCHOR_PHRASES, normalize_embeddings=True
            )
            self._anchor_vecs = vecs.astype(np.float32)
            log.info(
                f"[GraphJoinResolver] {len(_RELATIONAL_ANCHOR_PHRASES)} "
                "relational anchor embeddings ready."
            )
        except Exception as exc:
            log.warning(f"[GraphJoinResolver] Anchor embedding failed: {exc}")

    # ── column-word helper (reuses the chatbot's _TABLE_COL_WORDS logic) ──────
    _COL_STOPWORDS = {
        "id", "lov", "date", "time", "flag", "num", "no", "ref",
        "val", "code", "name", "desc", "the", "and", "for", "col", "row",
        "key", "set", "log",
    }

    def _col_words(self, table: str) -> set:
        words: set = set()
        for col in self._table_cols.get(table, set()):
            for tok in re.split(r'[_\s]+', col.lower()):
                if len(tok) >= 3 and tok not in self._COL_STOPWORDS:
                    words.add(tok)
        return words

    def _q_words(self, question: str) -> set:
        return set(re.findall(r'[a-z]{3,}', question.lower()))

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 2 — Semantic Score
    #   score = keyword_score + column_overlap_score + embedding_score
    #   if score > _JOIN_SCORE_THRESHOLD → proceed to Gate 3
    # ─────────────────────────────────────────────────────────────────────────
    def _semantic_score(self, question: str, tables: List[str]) -> float:
        score = 0.0
        q_words = self._q_words(question)

        # 2a. Keyword signal
        if _RELATIONAL_PHRASES.search(question):
            score += 1.0
            log.debug("[gate2] relational phrase detected +1.0")

        # 2b. Column overlap across ≥ 2 tables
        tables_with_col_hit = sum(
            1 for t in tables if q_words & self._col_words(t)
        )
        if tables_with_col_hit >= 2:
            score += 1.0
            log.debug(f"[gate2] col overlap in {tables_with_col_hit} tables +1.0")

        # 2c. Embedding similarity to relational anchor phrases
        if self._anchor_vecs is not None and self._embedder is not None:
            try:
                q_vec = self._embedder.encode(
                    question, normalize_embeddings=True
                ).astype(np.float32)
                sims     = self._anchor_vecs @ q_vec          # cosine (normalised)
                max_sim  = float(sims.max())
                emb_contrib = max_sim * _JOIN_EMB_WEIGHT
                score   += emb_contrib
                log.debug(f"[gate2] embedding sim={max_sim:.3f} contrib={emb_contrib:.3f}")
            except Exception as exc:
                log.warning(f"[gate2] embedding failed (non-fatal): {exc}")

        return score

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 3 — Neo4j weighted path (with BFS fallback)
    # ─────────────────────────────────────────────────────────────────────────
    def find_path(
        self,
        base:   str,
        target: str,
        intent: str = "simple",
    ) -> Optional[List[dict]]:
        """
        Return the cheapest join path from `base` to `target` as a list of
        step-dicts, or None if no path exists within depth 4.

        Cache key includes intent so that different query types on the same
        table pair can reuse the cheapest path independently.
        """
        cache_key = (base.upper(), target.upper(), intent)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        result: Optional[List[dict]] = None

        # ── Neo4j weighted Cypher ─────────────────────────────────────────────
        if self._neo4j_ok and self._driver:
            try:
                with self._driver.session() as sess:
                    rec = sess.run(
                        _CYPHER_WEIGHTED_PATH,
                        base=base, target=target,
                    ).single()
                    if rec:
                        result = _path_to_steps(rec["p"])
                        cost   = rec["cost"]
                        log.info(
                            f"[neo4j] path {base}→{target} "
                            f"({len(result)} hops, cost={cost})"
                        )
            except Exception as exc:
                log.warning(f"[neo4j] query failed ({exc}), falling back to BFS")

        # ── JSON-BFS fallback ─────────────────────────────────────────────────
        if result is None:
            result = self._bfs.find_path(base, target)
            if result:
                log.info(f"[bfs-fallback] path {base}→{target} ({len(result)} hops)")

        self._path_cache[cache_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # JOIN-NECESSITY CHECK
    # If every column word the question references lives in a single table,
    # there is no reason to join → force simple.
    # ─────────────────────────────────────────────────────────────────────────
    def _join_is_necessary(self, question: str, tables: List[str]) -> bool:
        """
        Returns False (join NOT necessary) when all question column-words are
        covered by a single table.
        Returns True when the question genuinely spans ≥ 2 tables.
        """
        q_words = self._q_words(question)
        required_cols = {
            w for t in tables for w in self._col_words(t) if w in q_words
        }
        if not required_cols:
            # No column signal at all — be conservative, assume single
            return False

        for table in tables:
            t_words = self._col_words(table)
            if required_cols <= t_words:
                log.info(
                    f"[necessity] All required cols found in '{table}' alone → simple"
                )
                return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: 3-GATE CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    def classify(
        self,
        question: str,
        tables:   List[str],
        intent:   str = "simple",
    ) -> str:
        """
        Returns "simple" or "relational".

        Gate 1 → single table → always "simple"
        Gate 2 → semantic score below threshold → "simple"
        Gate 3 → no Neo4j path exists → "simple"
        Gate 4 → join necessity check fails → "simple"
        All gates pass → "relational"
        """
        # ── Gate 1: trivially single ──────────────────────────────────────────
        if len(tables) <= 1:
            log.info("[classify] Gate1 → simple (single table)")
            return "simple"

        # ── Intents that must never join ──────────────────────────────────────
        _no_join_intents = {"aggregation", "group_by", "existence", "sort_by"}
        if intent in _no_join_intents:
            log.info(f"[classify] Gate1b → simple (intent={intent!r} blocks join)")
            return "simple"

        # ── Gate 2: semantic score ────────────────────────────────────────────
        score = self._semantic_score(question, tables)
        log.info(f"[classify] Gate2 semantic_score={score:.3f} threshold={_JOIN_SCORE_THRESHOLD}")
        if score < _JOIN_SCORE_THRESHOLD:
            log.info("[classify] Gate2 → simple (score below threshold)")
            return "simple"

        # ── Gate 3: Neo4j reachability for each target ────────────────────────
        base    = tables[0]
        targets = [t for t in tables[1:] if t.lower() != base.lower()]
        reachable = [t for t in targets if self.find_path(base, t, intent) is not None]
        if not reachable:
            log.info(f"[classify] Gate3 → simple (no path from '{base}' to any target)")
            return "simple"

        # ── Gate 4: join-necessity check ──────────────────────────────────────
        if not self._join_is_necessary(question, tables):
            log.info("[classify] Gate4 → simple (join not necessary)")
            return "simple"

        log.info(
            f"[classify] → RELATIONAL "
            f"(score={score:.3f}, reachable={reachable})"
        )
        return "relational"

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: BUILD JOIN CLAUSE
    # Drop-in replacement for build_join_clause() in chatbot_backend.py.
    # ─────────────────────────────────────────────────────────────────────────
    def build_join_clause(
        self,
        base:        str,
        tables:      List[str],
        intent:      str = "simple",
    ) -> str:
        """
        Build a SQL JOIN clause string starting from `base`.
        Uses weighted Neo4j paths (or BFS fallback).
        """
        clause, used = base, {base.lower()}

        for tgt in tables:
            if tgt.lower() == base.lower():
                continue
            path = self.find_path(base, tgt, intent)
            if path is None:
                log.warning(
                    f"[build_join] No path {base}→{tgt} — skipping"
                )
                continue
            for step in path:
                tt = step["to_table"]
                if tt.lower() not in used:
                    clause += (
                        f" INNER JOIN {tt}"
                        f" ON {step['from_table']}.{step['from_column']}"
                        f" = {tt}.{step['to_column']}"
                    )
                    used.add(tt.lower())

        return clause

    # ── status for /health endpoint ───────────────────────────────────────────
    def status(self) -> dict:
        return {
            "neo4j_connected":   self._neo4j_ok,
            "neo4j_uri":         _NEO4J_URI,
            "path_cache_entries": len(self._path_cache),
            "tables_indexed":    len(self._table_cols),
            "anchor_embeddings": (
                len(_RELATIONAL_ANCHOR_PHRASES)
                if self._anchor_vecs is not None else 0
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — run directly to verify join graph loading and Neo4j connectivity
# python core/graph_join_resolver.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n── Loading join graphs ──────────────────────────────────")
    jd = load_all_join_graphs()
    jp = jd.get("join_paths", {})
    print(f"  join_paths: {len(jp)} source tables")
    for tbl, targets in list(jp.items())[:5]:
        print(f"    {tbl} → {list(targets.keys())}")

    print("\n── Testing Neo4j connectivity ───────────────────────────")
    resolver = GraphJoinResolver(meta={}, join_graph=jd)
    ok = resolver.connect()
    print(f"  Neo4j connected: {ok}")
    if ok:
        print("  Status:", resolver.status())
    resolver.close()
    print("\nDone.")