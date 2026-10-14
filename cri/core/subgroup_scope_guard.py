from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("NL2SQL.SubgroupScopeGuard")

# ── Tunables — override via env vars ──────────────────────────────────────────
_MIN_SIGNAL_THRESHOLD: float = float(os.getenv("SCOPE_MIN_SIGNAL", "0.22"))
_LEEWAY: float               = float(os.getenv("SCOPE_LEEWAY",     "0.08"))  # compat only

# Queries at or below this word count get a relaxed threshold because
# short sentences produce weak embeddings regardless of topic.
_SHORT_QUERY_WORDS: int   = int(os.getenv("SCOPE_SHORT_QUERY_WORDS", "5"))
_SHORT_QUERY_BOOST: float = float(os.getenv("SCOPE_SHORT_QUERY_BOOST", "0.10"))

# Universal commands — "show all X", "list X", "get all X" — are always
# treated as in-scope when the noun X matches a known table/keyword in the
# selected subgroup.
_UNIVERSAL_RE = re.compile(
    r'^\s*(?:show|list|get|display|fetch|give\s+me|find)\s+(?:all|every|each)?\s*(.+)$',
    re.IGNORECASE,
)


# ── Module prefix helpers — env-driven, no hardcoded MFC_ anywhere ───────────

def _build_strip_prefixes() -> tuple:
    """
    Build tuple of module prefixes from env vars.
    Same logic as semantic_execution_planner so both stay in sync.

    Sources (first non-empty wins):
      1. TABLE_NAME_PREFIXES  e.g. "MFC_M_,MFC_L_,MFC_T_,ALM_M_,ALM_L_"
      2. TABLE_PREFIX         e.g. "MFC,ALM,USR"
         → expanded to "<CODE>_M_", "<CODE>_L_", "<CODE>_T_", "<CODE>_"
      3. Hard fallback        → ("MFC_M_", "MFC_L_", "MFC_T_", "MFC_")
    """
    raw_name = os.getenv("TABLE_NAME_PREFIXES", "").strip()
    if raw_name:
        prefixes = tuple(p.strip() for p in raw_name.split(",") if p.strip())
        if prefixes:
            return prefixes

    raw_code = os.getenv("TABLE_PREFIX", "").strip()
    if raw_code:
        codes = [c.strip().upper() for c in raw_code.split(",") if c.strip()]
        return tuple(
            f"{c}{sfx}"
            for c in codes
            for sfx in ("_M_", "_L_", "_T_", "_")
        )

    return ("MFC_M_", "MFC_L_", "MFC_T_", "MFC_")


_STRIP_PREFIXES: tuple = _build_strip_prefixes()
# Lower-cased version for anchor-term extraction (avoids repeated .lower() calls)
_STRIP_PREFIXES_LOWER: tuple = tuple(p.lower() for p in _STRIP_PREFIXES)


def _strip_table_prefix(name: str, lower: bool = False) -> str:
    """
    Strip any known module prefix from a table name.
    Longest prefix is tried first to prevent partial matches.
    If lower=True, input is assumed already lowercased (uses _STRIP_PREFIXES_LOWER).
    """
    prefixes = _STRIP_PREFIXES_LOWER if lower else _STRIP_PREFIXES
    name_cmp = name if lower else name.upper()
    for pfx in sorted(prefixes, key=len, reverse=True):
        pfx_cmp = pfx if lower else pfx.upper()
        if name_cmp.startswith(pfx_cmp):
            return name[len(pfx):]
    return name


# ── JSON-schema helpers (v1 bare-list  ↔  v2 {scope_hard_block_keywords, tables}) ──

def _iter_tables(subgroup_value) -> List[dict]:
    """Return the table list from either v1 (bare list) or v2 (dict) subgroup value."""
    if isinstance(subgroup_value, dict):
        return subgroup_value.get("tables", [])
    return subgroup_value   # v1 bare list


def _get_block_keywords(subgroup_value) -> List[str]:
    """Return the subgroup-level hard-block keyword list (empty for v1 JSON)."""
    if isinstance(subgroup_value, dict):
        return subgroup_value.get("scope_hard_block_keywords", [])
    return []


# ── Document builders ─────────────────────────────────────────────────────────

def _build_table_document(table_meta: dict) -> str:
    """Rich semantic snippet for ONE table."""
    parts: List[str] = []

    name = table_meta.get("table_name", "")
    if name:
        human = _strip_table_prefix(name).replace("_", " ").lower()
        parts.append(human)

    for key in ("business_purpose", "row_represents"):
        val = table_meta.get(key, "")
        if val:
            parts.append(val)

    synonyms: List[str] = table_meta.get("synonyms", [])
    if synonyms:
        parts.append(". ".join(synonyms))

    use_when: List[str] = table_meta.get("use_when", [])
    if use_when:
        parts.append(". ".join(use_when))

    do_not_use: List[str] = table_meta.get("do_not_use_when", [])
    if do_not_use:
        parts.append("Not for: " + ". ".join(do_not_use))

    return " ".join(filter(None, parts))


def _build_subgroup_document(tables: List[dict]) -> str:
    """
    ONE document per subgroup — all table snippets concatenated.
    The combined vector captures the full semantic surface of everything
    answerable in this subgroup.
    """
    snippets = [_build_table_document(t) for t in tables if t.get("table_name")]
    return " | ".join(filter(None, snippets))


def _build_anchor_terms(tables: List[dict]) -> set:
    """
    Build a flat set of lowercased keywords extracted from every table name
    in the subgroup.  Used for the universal-command fast-pass check so that
    "show all activity" immediately matches a subgroup that contains an
    Activity table without needing cosine similarity at all.

    e.g. MFC_M_Activity → {"activity"}
         MFC_L_Event_Type → {"event", "type"}
    """
    terms: set = set()
    for t in tables:
        name = t.get("table_name", "").lower()
        name = _strip_table_prefix(name, lower=True)
        for tok in name.split("_"):
            if len(tok) >= 3:
                terms.add(tok)
        # also add synonyms if present
        for syn in t.get("synonyms", []):
            for tok in re.findall(r'[a-z]{3,}', syn.lower()):
                terms.add(tok)
    return terms


# ── Guard ─────────────────────────────────────────────────────────────────────

class SubgroupScopeGuard:
    """
    Scope guard: one embedding per subgroup (aggregating all its tables).

    Two-stage validation:
      Stage 1 — Universal-command fast-pass
        "show all X / list X / get all X" where X matches a table keyword
        in the subgroup → immediately pass.  Avoids penalising the most
        common short query pattern which always scores low on cosine sim.

      Stage 2 — Cosine similarity with adaptive threshold
        Short queries (≤ SCOPE_SHORT_QUERY_WORDS words) get a boosted
        threshold tolerance (threshold lowered by SCOPE_SHORT_QUERY_BOOST)
        because short sentence embeddings are naturally less specific and
        will land closer to the global centroid than domain-specific ones.

    Parameters
    ----------
    module_data : dict
        Full content of mfc_narrow.json ({"modules": {...}}).
    embedder : SentenceTransformer
        Shared embedder instance from chatbot_backend.py.
    min_signal : float
        Base rejection threshold (cosine similarity).
    leeway : float
        Unused; kept for env-var / health-endpoint compatibility.
    """

    def __init__(
        self,
        module_data: dict,
        embedder,
        min_signal: float = _MIN_SIGNAL_THRESHOLD,
        leeway:     float = _LEEWAY,
    ) -> None:
        self._embedder   = embedder
        self._min_signal = min_signal
        self._leeway     = leeway   # compat

        # (module_key, subgroup_key) → {vector, sg_label, mod_label,
        #                               table_names, anchor_terms}
        self._subgroup_index: Dict[Tuple[str, str], dict] = {}
        self._build_index(module_data)

        log.info(
            f"[SubgroupScopeGuard] Ready — {len(self._subgroup_index)} subgroup "
            f"embedding(s) built."
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_index(self, module_data: dict) -> None:
        modules = module_data.get("modules", {})

        keys:  List[Tuple[str, str]] = []
        docs:  List[str]             = []
        metas: List[dict]            = []

        for mod_key, mod_val in modules.items():
            mod_label = mod_val.get("module_label", mod_key)
            for sg_key, sg_val in mod_val.get("subgroups", {}).items():
                tables = _iter_tables(sg_val)
                if not tables:
                    continue
                doc = _build_subgroup_document(tables)
                if not doc.strip():
                    continue

                # Pre-compile hard-block keyword patterns (v2 JSON only; empty for v1)
                raw_kws = _get_block_keywords(sg_val)
                block_patterns: List[Tuple[str, re.Pattern]] = []
                for kw in raw_kws:
                    kw_norm = kw.lower().strip()
                    # Word-boundary for single-word tokens; plain substring for phrases
                    if " " in kw_norm:
                        pat = re.compile(re.escape(kw_norm), re.IGNORECASE)
                    else:
                        pat = re.compile(r"\b" + re.escape(kw_norm) + r"\b", re.IGNORECASE)
                    block_patterns.append((kw_norm, pat))

                keys.append((mod_key, sg_key))
                docs.append(doc)
                metas.append({
                    "mod_label":      mod_label,
                    "sg_label":       sg_key.replace("_", " ").title(),
                    "table_names":    [
                        t["table_name"] for t in tables if t.get("table_name")
                    ],
                    "anchor_terms":   _build_anchor_terms(tables),
                    "block_patterns": block_patterns,
                })

        if not docs:
            log.warning("[SubgroupScopeGuard] No subgroups found in module_data.")
            return

        log.info(f"[SubgroupScopeGuard] Embedding {len(docs)} subgroup document(s) …")
        vectors = self._embedder.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )

        for i, key in enumerate(keys):
            self._subgroup_index[key] = {**metas[i], "vector": vectors[i]}

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )

    def _keyword_blocked(self, question: str, block_patterns: List[Tuple[str, re.Pattern]]) -> Optional[str]:
        """
        Return the first hard-block keyword that matches *question*, or None.
        Called before any embedding work — O(n_keywords) string scan only.
        """
        for kw_norm, pat in block_patterns:
            if pat.search(question):
                return kw_norm
        return None

    def _universal_command_match(self, question: str, anchor_terms: set) -> bool:
        """
        Return True when the question is a universal command
        ("show all X", "list X", "get all X", etc.) and the noun X
        overlaps with at least one anchor term of the subgroup.

        This bypasses cosine similarity entirely for the most common
        short queries like "show all activity" or "list all events".
        """
        m = _UNIVERSAL_RE.match(question.strip())
        if not m:
            return False
        # Extract meaningful words from the noun phrase
        noun_phrase = m.group(1).lower()
        noun_words  = {w for w in re.findall(r'[a-z]{3,}', noun_phrase)}
        hit = bool(noun_words & anchor_terms)
        if hit:
            log.info(
                f"[scope_guard] FAST-PASS (universal command) — "
                f"noun_words={noun_words} ∩ anchor_terms={noun_words & anchor_terms}"
            )
        return hit

    # ── public ────────────────────────────────────────────────────────────────

    def _fuzzy_lookup(
        self, module_name: str, subgroup_name: str
    ) -> "Tuple[Tuple[str,str], Optional[dict]]":
        """
        Try to resolve (module_name, subgroup_name) to a real index key when
        an exact match fails.  Handles two common mismatches:

          • Singular/plural drift  : "user_profile" ↔ "user_profiles"
          • Missing connector words: "user_org_masters" ↔ "user_and_org_masters"

        Strategy: score every indexed key by the number of underscore-tokens
        from (module_name + subgroup_name) that appear in (mod_key + sg_key).
        Returns the best-scoring key if it shares ≥ 50 % of the query tokens,
        otherwise returns (original_key, None).
        """
        query_tokens = set(
            re.findall(r'[a-z0-9]+', f"{module_name}_{subgroup_name}".lower())
        )
        if not query_tokens:
            return (module_name, subgroup_name), None

        best_key:   Optional[Tuple[str, str]] = None
        best_score: float = 0.0

        for (mod_k, sg_k) in self._subgroup_index:
            candidate_tokens = set(
                re.findall(r'[a-z0-9]+', f"{mod_k}_{sg_k}".lower())
            )
            if not candidate_tokens:
                continue
            overlap = len(query_tokens & candidate_tokens)
            # Jaccard-style score
            score = overlap / len(query_tokens | candidate_tokens)
            if score > best_score:
                best_score = score
                best_key   = (mod_k, sg_k)

        # Require at least 50 % overlap to avoid nonsense matches
        if best_key and best_score >= 0.50:
            return best_key, self._subgroup_index[best_key]

        return (module_name, subgroup_name), None

    @property
    def _names(self) -> List[str]:
        """Flat list of all table names — used by /health endpoint."""
        names: List[str] = []
        for meta in self._subgroup_index.values():
            names.extend(meta["table_names"])
        return names

    def check(
        self,
        question:      str,
        module_name:   str,
        subgroup_name: str,
    ) -> dict:
        """
        Full reverse-lookup scope check.

        Scores the question against EVERY subgroup in the index to find
        the best match, then decides whether the selected subgroup is
        in-scope.

        Returns
        -------
        dict with keys:
            in_scope        bool   — True if the question belongs here
            best_subgroup   str    — subgroup key of the top-scoring match
            best_module     str    — module key of the top-scoring match
            best_sg_label   str    — human-readable label of best match
            best_score      float  — cosine similarity of best match
            selected_score  float  — cosine similarity of the selected subgroup
            fast_pass       bool   — True when the universal-command path fired
        """
        key      = (module_name, subgroup_name)
        sg_entry = self._subgroup_index.get(key)

        # ── Key-miss recovery ─────────────────────────────────────────────────
        # The frontend may send concept narrow_keys that don't exactly match the
        # narrow JSON module/subgroup keys (e.g. "user_profile" vs "user_profiles",
        # "user_org_masters" vs "user_and_org_masters").
        # Try to find the closest match before giving up and skipping the check.
        if sg_entry is None:
            resolved_key, sg_entry = self._fuzzy_lookup(module_name, subgroup_name)
            if sg_entry is not None:
                log.debug(
                    "[SubgroupScopeGuard] Fuzzy-matched '%s/%s' → '%s/%s'",
                    module_name, subgroup_name, resolved_key[0], resolved_key[1],
                )
                key           = resolved_key
                module_name   = resolved_key[0]
                subgroup_name = resolved_key[1]

        # Unknown subgroup → always pass (don't block if config is incomplete)
        if sg_entry is None:
            log.warning(
                "[SubgroupScopeGuard] Unknown subgroup '%s/%s' — check skipped.",
                module_name, subgroup_name,
            )
            return {
                "in_scope":       True,
                "best_subgroup":  subgroup_name,
                "best_module":    module_name,
                "best_sg_label":  subgroup_name,
                "best_score":     1.0,
                "selected_score": 1.0,
                "fast_pass":      False,
            }

        # ── Gate 0: keyword hard-block ────────────────────────────────────────
        # Runs embedding to find the best alternative for a meaningful redirect message.
        kw_hit = self._keyword_blocked(question, sg_entry["block_patterns"])
        if kw_hit is not None:
            log.info(
                "[scope_guard] KEYWORD BLOCK | subgroup=%s/%s | matched='%s' | q='%s'",
                module_name, subgroup_name, kw_hit, question[:80],
            )
            # Run embedding scoring to find the best-matching alternative subgroup
            # so the caller can show a meaningful redirect (e.g. "belongs to Serial Master").
            q_vec = self._embed(question)
            kw_best_score: float = -1.0
            kw_best_key = (module_name, subgroup_name)
            for (mod_k, sg_k), entry in self._subgroup_index.items():
                if mod_k == module_name and sg_k == subgroup_name:
                    continue   # skip the blocked subgroup itself
                score = float(entry["vector"] @ q_vec)
                if score > kw_best_score:
                    kw_best_score = score
                    kw_best_key   = (mod_k, sg_k)
            kw_best_entry = self._subgroup_index[kw_best_key]
            kw_best_mod, kw_best_sg = kw_best_key
            return {
                "in_scope":       False,
                "best_subgroup":  kw_best_sg,
                "best_module":    kw_best_mod,
                "best_sg_label":  kw_best_entry["sg_label"],
                "best_score":     kw_best_score,
                "selected_score": 0.0,
                "fast_pass":      False,
                "keyword_hit":    kw_hit,          # extra key — callers may inspect
            }

        # ── Stage 1: universal-command fast-pass ─────────────────────────────
        if self._universal_command_match(question, sg_entry["anchor_terms"]):
            return {
                "in_scope":       True,
                "best_subgroup":  subgroup_name,
                "best_module":    module_name,
                "best_sg_label":  sg_entry["sg_label"],
                "best_score":     1.0,
                "selected_score": 1.0,
                "fast_pass":      True,
            }

        # ── Stage 2: score question against ALL subgroups ─────────────────────
        q_vec = self._embed(question)

        best_score:    float = -1.0
        best_key:      Tuple[str, str] = key
        selected_score: float = -1.0

        for (mod_k, sg_k), entry in self._subgroup_index.items():
            score = float(entry["vector"] @ q_vec)
            if mod_k == module_name and sg_k == subgroup_name:
                selected_score = score
            if score > best_score:
                best_score = score
                best_key   = (mod_k, sg_k)

        best_entry = self._subgroup_index[best_key]
        best_mod, best_sg = best_key

        # Adaptive threshold for short queries
        word_count          = len(question.strip().split())
        effective_threshold = self._min_signal
        if word_count <= _SHORT_QUERY_WORDS:
            effective_threshold = max(0.0, self._min_signal - _SHORT_QUERY_BOOST)

        # In-scope when:
        #   (a) selected subgroup IS the best match, OR
        #   (b) selected subgroup is within leeway of the best match, OR
        #   (c) the question doesn't meet min_signal for ANY subgroup
        #       (too generic — pass rather than give a confusing redirect)
        in_scope = (
            best_key == key                                        # (a) exact best
            or (selected_score >= best_score - self._leeway)      # (b) within leeway
            or (best_score < effective_threshold)                  # (c) generic/weak
        )

        log.info(
            "[scope_guard] check | q='%s' | selected=%s/%s(%.4f) | "
            "best=%s/%s(%.4f) | leeway=%.2f | in_scope=%s",
            question[:60],
            module_name, subgroup_name, selected_score,
            best_mod, best_sg, best_score,
            self._leeway, in_scope,
        )

        return {
            "in_scope":       in_scope,
            "best_subgroup":  best_sg,
            "best_module":    best_mod,
            "best_sg_label":  best_entry["sg_label"],
            "best_score":     best_score,
            "selected_score": selected_score,
            "fast_pass":      False,
        }

    def validate(
        self,
        question:      str,
        module_name:   str,
        subgroup_name: str,
    ) -> Tuple[bool, str]:
        """
        Returns (True, "") if the question is relevant to the selected subgroup,
        or (False, reason) with a user-facing explanation if it is not.

        Delegates to check() so both methods share the same logic.
        The reason string now includes which subgroup the question belongs to,
        making it actionable for the user.
        """
        result = self.check(
            question=question,
            module_name=module_name,
            subgroup_name=subgroup_name,
        )

        if result["in_scope"]:
            return True, ""

        # ── Build user-facing rejection message ───────────────────────────────
        # If the query was hard-blocked by keyword, give a tighter message.
        kw_hit = result.get("keyword_hit")
        if kw_hit:
            key      = (module_name, subgroup_name)
            sg_entry = self._subgroup_index.get(key)
            selected_sg_label = sg_entry["sg_label"] if sg_entry else subgroup_name
            reason = (
                f"Your question mentions **'{kw_hit}'**, which is out of scope for "
                f"**{selected_sg_label}**. Please switch to the relevant module or "
                f"ask something related to {selected_sg_label}."
            )
            log.info(
                "[scope_guard] REJECT (keyword) | subgroup='%s/%s' | kw='%s'",
                module_name, subgroup_name, kw_hit,
            )
            return False, reason
        key      = (module_name, subgroup_name)
        sg_entry = self._subgroup_index.get(key)

        _humanise = lambda t: _strip_table_prefix(t).replace("_", " ").title()

        selected_sg_label = sg_entry["sg_label"] if sg_entry else subgroup_name
        tables            = sg_entry["table_names"] if sg_entry else []
        table_hint        = ", ".join(_humanise(t) for t in tables[:5])
        if len(tables) > 5:
            table_hint += f" and {len(tables) - 5} more"

        best_sg_label = result["best_sg_label"]
        best_mod      = result["best_module"]

        # Only mention the detected subgroup if it's different from selected
        # and has a meaningful score (avoids confusing redirects for generic queries)
        if best_sg_label != selected_sg_label and result["best_score"] >= self._min_signal:
            redirect_hint = (
                f" It looks like a **{best_sg_label}** query "
                f"(module: {best_mod}). "
                f"Please switch to the correct module/subgroup, or"
            )
        else:
            redirect_hint = " Please"

        reason = (
            f"Your question doesn't seem to be about **{selected_sg_label}**. "
            f"This section covers: {table_hint}.{redirect_hint} "
            f"ask something related to {selected_sg_label}."
        )

        log.info(
            "[scope_guard] REJECT | subgroup='%s/%s' | "
            "selected_score=%.4f | best='%s/%s'(%.4f)",
            module_name, subgroup_name,
            result["selected_score"],
            best_mod, result["best_subgroup"], result["best_score"],
        )
        return False, reason

    def describe_subgroup(self, module_name: str, subgroup_name: str) -> str:
        """Human-readable subgroup summary for frontend tooltips."""
        entry = self._subgroup_index.get((module_name, subgroup_name))
        if not entry:
            return ""
        return (
            f"{entry['sg_label']} covers "
            f"{len(entry['table_names'])} table(s): "
            f"{', '.join(entry['table_names'])}."
        )