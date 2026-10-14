from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("NL2SQL.GeneralGuard")

# ── Tunables — override via env vars ──────────────────────────────────────────
# Minimum cosine similarity between the question and the MES domain anchor.
# Questions scoring below this are considered off-topic.
_MES_MIN_SCORE: float = float(os.getenv("GENERAL_GUARD_MIN_SCORE", "0.13"))

# Very short questions (≤ this word count) get a relaxed threshold because
# single-word / two-word embeddings are naturally close to everything.
_SHORT_QUERY_WORDS: int   = int(os.getenv("GENERAL_GUARD_SHORT_WORDS", "4"))
_SHORT_QUERY_BOOST: float = float(os.getenv("GENERAL_GUARD_SHORT_BOOST", "0.08"))


# ── MES domain anchor sentences ───────────────────────────────────────────────
# These describe the full semantic space of the MES system.
# The embedder averages them into a single domain vector at startup.
_MES_ANCHOR_SENTENCES: List[str] = [
    # Keyword-dense for broad semantic coverage
    "manufacturing execution system work orders production shop floor",
    "bill of materials BOM routing operations components assembly",
    "serial number serialization tracking history traceability lot batch",
    "quality inspection defect non-conformance sampling plan",
    "item master material product code inventory stock",
    "work order execution activity scheduling operator machine workstation",
    "device management calibration maintenance equipment",
    "alarm management notification alert threshold",
    "tenant organization configuration setup master data",
    "engineering BOM manufacturing BOM revision effectivity",
    "packet template document attachment work instruction",
    "lot batch number production order status tracking",
    "shift production reporting yield scrap rework",
    "supplier vendor purchase order goods receipt",
    # Natural-language phrasings that mirror real user queries
    "what routing lines are assigned to this routing?",
    "which operations are part of this routing?",
    "show me the work orders for this production order",
    "what is the status of this work order?",
    "list all serial numbers for this item",
    "which components are in this bill of materials?",
    "what defects were found during inspection?",
    "show the BOM for this product",
    "which machine or workstation is assigned to this operation?",
    "what items are produced in this lot?",
    "show production schedule for this work center",
    "what is the yield and scrap for this work order?",
    "list all non-conformances raised today",
    "which supplier provided this material?",
    "what routing line got assigned to this routing?",
    "show all operations under this routing",
]


# ── SQL injection / raw query block ──────────────────────────────────────────
# Detects user inputs that ARE SQL queries rather than natural language questions.
# Triggers if the input starts with a SQL keyword OR contains a full SQL clause.
_SQL_QUERY_RE: re.Pattern = re.compile(
    # ── Full statement starters ───────────────────────────────────────────────
    r'^\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|'
    r'WITH|MERGE|REPLACE|CALL|EXPLAIN)\b'
    r'|'
    # ── Full clause shapes ────────────────────────────────────────────────────
    r'\b(?:SELECT\s+[\w\*]|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|'
    r'DROP\s+(?:TABLE|DATABASE|INDEX|VIEW|COLUMN)|ALTER\s+TABLE|'
    r'CREATE\s+(?:TABLE|INDEX|VIEW|DATABASE)|TRUNCATE\s+TABLE|'
    r'EXEC(?:UTE)?\s+\w|FROM\s+\w+\s+WHERE|JOIN\s+\w+\s+ON)\b'
    r'|'
    # ── Fragment / injection patterns ─────────────────────────────────────────
    # WHERE clause fragment: "WHERE id=1", "where id = 2"
    r'\bWHERE\s+\w+\s*[=<>!]+\s*[\w\'\"]'
    r'|'
    # Classic quote injection: "' OR '1'='1", "' AND '1'='1"
    r"'\s*(?:OR|AND)\s*'"
    r'|'
    # Tautology / always-true injections: "1=1", "1=2", "'a'='a'"
    r'(?:\'[^\']*\'|\b\d+)\s*=\s*(?:\'[^\']*\'|\d+)'
    r'|'
    # OR/AND injection with digits or quoted strings
    r'\b(?:OR|AND)\s+(?:\'[^\']*\'|\d+)\s*=\s*(?:\'[^\']*\'|\d+)'
    r'|'
    # Comment injection: "--", "/*", "*/"
    r'(?:--|/\*|\*/)'
    r'|'
    # UNION injection
    r'\bUNION\s+(?:ALL\s+)?SELECT\b'
    r'|'
    # Stacked query: semicolon followed by SQL keyword
    r';\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|EXEC)\b'
    r'|'
    # SQL comparison operators in suspicious context: "id=1", "col >= 2"
    r'\b\w+\s*(?:=|!=|<>|>=|<=)\s*\d+\b'
    r'|'
    # SQL functions commonly used in injection
    r'\b(?:SLEEP|BENCHMARK|WAITFOR\s+DELAY|LOAD_FILE|INTO\s+OUTFILE|'
    r'INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS|XP_CMDSHELL)\b',
    re.IGNORECASE,
)


# ── Hard-block patterns — rejected instantly, no embedding needed ─────────────
# Each tuple: (human-readable name, compiled regex)
_HARD_BLOCK_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "greeting",
        re.compile(
            r'^\s*(?:hi+|hey+|hello+|good\s+(?:morning|afternoon|evening|night)|'
            r'howdy|greetings|sup|what\'?s\s+up)'
            r'(?:\s+(?:there|everyone|all|team|folks|claude))?\s*[!?.]?\s*$',
            re.IGNORECASE,
        ),
    ),
    (
        "bot identity",
        re.compile(
            r'\b(?:who\s+are\s+you|what\s+are\s+you|your\s+name|'
            r'who\s+made\s+you|who\s+built\s+you|who\s+created\s+you|'
            r'are\s+you\s+(?:an?\s+)?(?:ai|bot|robot|human|gpt|claude))\b',
            re.IGNORECASE,
        ),
    ),
    (
        "general knowledge",
        re.compile(
            r'\b(?:capital\s+of|president\s+of|prime\s+minister\s+of|'
            r'population\s+of|currency\s+of|language\s+of|'
            r'largest\s+country|smallest\s+country|'
            r'who\s+(?:won|invented|discovered|wrote|founded)|'
            r'when\s+(?:was|did|were)\s+(?!.*(?:order|bom|serial|work|production|item)))'
            r'\b',
            re.IGNORECASE,
        ),
    ),
    (
        "entertainment / lifestyle",
        re.compile(
            r'\b(?:movie|film|song|music|cricket|football|soccer|basketball|'
            r'tennis|recipe|cook|food|restaurant|weather|forecast|'
            r'horoscope|joke|funny|meme|celebrity|actor|actress|'
            r'news|politics|election|stock\s+market|crypto|bitcoin)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "pure math / trivia",
        re.compile(
            r'^\s*(?:what\s+is\s+)?(?:\d+\s*[\+\-\*\/\^]\s*\d+|'
            r'square\s+root|factorial|fibonacci|prime\s+number)\s*\??\s*$',
            re.IGNORECASE,
        ),
    ),
    (
        "personal / emotional",
        re.compile(
            r'\b(?:i\s+(?:love|hate|feel|am\s+(?:sad|happy|angry|bored|tired|lonely))|'
            r'my\s+(?:life|relationship|girlfriend|boyfriend|wife|husband|family)|'
            r'help\s+me\s+with\s+my\s+(?:homework|essay|resume|cv))\b',
            re.IGNORECASE,
        ),
    ),
    (
        "numeric only",
        re.compile(
            # Entire input is digits, decimals, commas, spaces — no real words
            r'^\s*[\d\s,.\-]+\s*$',
        ),
    ),
    (
        "greeting with name / bare name",
        re.compile(
            # Matches: "hey shree", "hi narmatha", "hello john doe",
            # or a bare name/word dump (1-3 words) that contains NO action
            # verb at the start (show/list/get/find/what/which/how/give etc.)
            # Uses a negative-lookahead so "show work orders" never fires here.
            r'^\s*(?:(?:hi+|hey+|hello+|good\s+(?:morning|afternoon|evening|night))\s+)'
            r'(?!(?:show|list|get|find|what|which|how|give|tell|display|fetch|'
            r'count|sum|check|describe|explain|create|update|delete|select|where)\b)'
            r'[A-Za-z]{2,25}(?:\s+[A-Za-z]{2,25}){0,2}\s*[!.,]?\s*$'
            r'|'
            # Bare input (no greeting prefix): only block if it's 1-2 plain
            # words and NOT starting with an action verb or MES term
            r'^\s*(?!(?:show|list|get|find|what|which|how|give|tell|display|fetch|'
            r'count|sum|check|describe|explain|create|update|delete|select|where|'
            r'work|serial|bom|lot|batch|routing|defect|item|order|production|shift|'
            r'yield|scrap|machine|sensor|alarm|supplier|vendor|packet|assembly)\b)'
            r'[A-Za-z]{2,25}(?:\s+[A-Za-z]{2,25})?\s*[!.,]?\s*$',
            re.IGNORECASE,
        ),
    ),
]


# ── MES keyword allowlist — fast-pass before embedding ────────────────────────
# Any query containing a known MES term is immediately allowed, bypassing the
# embedding stage entirely.  This prevents false-negatives on valid but terse
# or unusually-phrased MES queries (e.g. "CNC Turning Routing").
_MES_ALLOWLIST_RE: re.Pattern = re.compile(
    r'\b(?:work\s*order|bom|bill\s+of\s+material|routing|serial\s+number|'
    r'serial|lot\s+number|lot|batch|production\s+order|work\s+center|'
    r'operation|component|assembly|inspection|defect|non.?conformance|'
    r'item\s+master|material|inventory|calibration|maintenance|'
    r'shift|yield|scrap|rework|supplier|purchase\s+order|packet|'
    r'traceability|effectivity|revision|cnc|machine|workstation)\b',
    re.IGNORECASE,
)


# ── Keyboard row / special character patterns ────────────────────────────────
# Detects keyboard walks (qwerty rows) and special-character noise.

# Known keyboard rows — any 4+ consecutive chars from the same row = gibberish
_KB_ROWS: list = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "1234567890",
]

# Special characters that have no place in a natural language MES query
_SPECIAL_CHARS_RE: re.Pattern = re.compile(r'[!@#$%^&*()\[\]{}<>|\\~`,.:;\'"\/\-_+=?]')

# Input that is entirely special characters / punctuation / brackets (no alphanumeric)
_SPECIAL_ONLY_RE: re.Pattern = re.compile(r'^[^a-zA-Z0-9\s]+$')

# Minimum ratio of special chars to total chars to flag as noise
_SPECIAL_CHAR_RATIO_MAX: float = float(os.getenv("GENERAL_GUARD_SPECIAL_RATIO", "0.30"))


def _has_keyboard_walk(text: str, min_run: int = 4) -> bool:
    """Return True if *text* contains a run of ≥ min_run consecutive keyboard-row chars."""
    lower = text.lower().replace(" ", "")
    for row in _KB_ROWS:
        # Forward walk
        run = 1
        for i in range(1, len(lower)):
            if lower[i] in row and lower[i - 1] in row:
                j_cur  = row.index(lower[i])
                j_prev = row.index(lower[i - 1])
                if abs(j_cur - j_prev) == 1:
                    run += 1
                    if run >= min_run:
                        return True
                else:
                    run = 1
            else:
                run = 1
    return False
# Min ratio of alphabetic chars that must be vowels (a e i o u y).
# Real English words have ~35-40 % vowels; gibberish like "fgfg" / "hjhj" has ~0 %.
_GIBBERISH_VOWEL_RATIO_MIN: float = float(os.getenv("GENERAL_GUARD_VOWEL_RATIO", "0.15"))

# Max ratio of digits to total characters — e.g. "234fgfg" is 43 % digits.
_GIBBERISH_DIGIT_RATIO_MAX: float = float(os.getenv("GENERAL_GUARD_DIGIT_RATIO", "0.40"))

# Compiled pattern: token looks like alphanumeric noise when it has ≥ 2
# consecutive consonant pairs AND at least one digit mixed in.
_GIBBERISH_TOKEN_RE: re.Pattern = re.compile(
    r'^[bcdfghjklmnpqrstvwxyz]{2,}[\d]+[bcdfghjklmnpqrstvwxyz\d]*$'
    r'|^[\d]+[bcdfghjklmnpqrstvwxyz]{2,}[\d\w]*$',
    re.IGNORECASE,
)

# If ALL tokens in the query look like gibberish tokens, block.
_GIBBERISH_CONSONANT_RE: re.Pattern = re.compile(
    r'^[bcdfghjklmnpqrstvwxyz]{3,}$',
    re.IGNORECASE,
)

_CONSONANT_ONLY_RE: re.Pattern = re.compile(
    r'^[bcdfghjklmnpqrstvwxyz]+$',
    re.IGNORECASE,
)


def _is_gibberish(text: str) -> bool:
    """
    Return True if *text* looks like keyboard noise / gibberish.

    Heuristics applied (all must agree to avoid false positives):
    1. Mixed alphanumeric tokens — letters and digits jumbled (e.g. "234fgfg").
    2. Very low vowel ratio across the entire query (< 15 %).
    3. All tokens are pure consonant clusters with no recognisable word shape.
    """
    stripped = text.strip()
    if not stripped:
        return False

    tokens = stripped.split()
    total_chars = len(stripped.replace(" ", ""))
    if total_chars == 0:
        return False

    alpha_chars  = [c for c in stripped if c.isalpha()]
    digit_chars  = [c for c in stripped if c.isdigit()]
    vowel_chars  = [c for c in alpha_chars if c.lower() in "aeiouy"]

    digit_ratio  = len(digit_chars) / total_chars
    vowel_ratio  = len(vowel_chars) / len(alpha_chars) if alpha_chars else 0.0

    # Rule 1: mixed alnum tokens (e.g. "234fgfg", "hjhj343")
    noise_tokens = [t for t in tokens if _GIBBERISH_TOKEN_RE.match(t)]
    if noise_tokens and len(noise_tokens) == len(tokens):
        return True

    # Rule 2: high digit ratio + low vowel ratio
    if digit_ratio >= _GIBBERISH_DIGIT_RATIO_MAX and vowel_ratio < _GIBBERISH_VOWEL_RATIO_MIN:
        return True

    # Rule 3: all tokens are consonant-only clusters (e.g. "hjhj", "fgfg")
    consonant_only = [t for t in tokens if _CONSONANT_ONLY_RE.match(t)]
    if consonant_only and len(consonant_only) == len(tokens):
        return True

    # Rule 4: very low vowel ratio across the whole query (no digits needed)
    if len(alpha_chars) >= 4 and vowel_ratio < _GIBBERISH_VOWEL_RATIO_MIN:
        return True

    # Rule 5: keyboard row walk (e.g. "qwertyuiop[]", "asdfgh")
    if _has_keyboard_walk(stripped):
        return True

    # Rule 6: input is entirely special characters / punctuation (e.g. "!@#$%^&*()")
    if _SPECIAL_ONLY_RE.match(stripped):
        return True

    # Rule 7: high ratio of special characters mixed with alphanumeric noise
    special_count = len(_SPECIAL_CHARS_RE.findall(stripped))
    if total_chars > 0 and (special_count / total_chars) >= _SPECIAL_CHAR_RATIO_MAX:
        return True

    return False


class GeneralGuard:
    """
    Pre-filter that rejects questions unrelated to the MES system
    before they reach the scope guard or the LLM pipeline.

    Two-stage check
    ---------------
    Stage 1 — Hard keyword block (O(patterns), no embedding)
        Instantly rejects obvious off-topic inputs: greetings, general
        knowledge questions, entertainment topics, etc.

    Stage 2 — MES domain embedding similarity
        Embeds the question and compares it against a pre-built MES
        domain anchor vector.  Questions whose cosine similarity falls
        below ``min_score`` are rejected as off-topic.
        Short queries (≤ SHORT_QUERY_WORDS words) get a relaxed
        threshold to avoid false-positives on terse but valid queries
        like "show BOMs" or "list serials".

    Parameters
    ----------
    embedder : SentenceTransformer
        Shared embedder instance (same one used by SubgroupScopeGuard).
    min_score : float
        Base cosine-similarity threshold below which a question is
        considered off-topic.  Default: env GENERAL_GUARD_MIN_SCORE (0.20).
    """

    def __init__(
        self,
        embedder,
        min_score: float = _MES_MIN_SCORE,
    ) -> None:
        self._embedder  = embedder
        self._min_score = min_score
        self._anchor_vec: np.ndarray = self._build_anchor()
        log.info(
            "[GeneralGuard] Ready — min_score=%.2f | anchor sentences=%d | "
            "hard-block rules=%d",
            self._min_score,
            len(_MES_ANCHOR_SENTENCES),
            len(_HARD_BLOCK_RULES),
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_anchor(self) -> np.ndarray:
        """Embed all anchor sentences and return their normalised mean vector."""
        vecs = self._embedder.encode(
            _MES_ANCHOR_SENTENCES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    def _hard_blocked(self, question: str) -> Optional[str]:
        """Return the rule name if any hard-block pattern matches, else None."""
        for rule_name, pat in _HARD_BLOCK_RULES:
            if pat.search(question):
                return rule_name
        return None

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )

    # ── public ────────────────────────────────────────────────────────────────

    def check(self, question: str) -> dict:
        """
        Check whether *question* is relevant to the MES domain.

        Returns
        -------
        dict with keys:
            allowed       bool   — True if the question should proceed
            reason        str    — 'allowed' | 'hard_block' | 'low_similarity'
            rule          str    — which hard-block rule fired (if any)
            score         float  — cosine similarity against MES anchor
                                   (0.0 if hard-blocked before embedding)
            threshold     float  — effective threshold used
        """
        q = question.strip()

        # ── Stage 0: raw SQL / injection block (before allowlist) ────────────
        if _SQL_QUERY_RE.search(q):
            log.info("[GeneralGuard] SQL BLOCK | q='%s'", q[:80])
            return {
                "allowed":   False,
                "reason":    "sql_query",
                "rule":      "sql_query",
                "score":     0.0,
                "threshold": self._min_score,
            }

        # ── Stage 0.5: MES keyword fast-pass ────────────────────────────────
        if _MES_ALLOWLIST_RE.search(q):
            log.info("[GeneralGuard] FAST PASS (MES keyword) | q='%s'", q[:80])
            return {
                "allowed":   True,
                "reason":    "allowed",
                "rule":      None,
                "score":     1.0,
                "threshold": self._min_score,
            }

        # ── Stage 1: gibberish detection ─────────────────────────────────────
        if _is_gibberish(q):
            log.info("[GeneralGuard] GIBBERISH BLOCK | q='%s'", q[:80])
            return {
                "allowed":   False,
                "reason":    "gibberish",
                "rule":      "gibberish",
                "score":     0.0,
                "threshold": self._min_score,
            }

        # ── Stage 1: hard keyword block ───────────────────────────────────────
        rule = self._hard_blocked(q)
        if rule is not None:
            log.info(
                "[GeneralGuard] HARD BLOCK | rule='%s' | q='%s'",
                rule, q[:80],
            )
            return {
                "allowed":   False,
                "reason":    "hard_block",
                "rule":      rule,
                "score":     0.0,
                "threshold": self._min_score,
            }

        # ── Stage 2: embedding similarity ─────────────────────────────────────
        word_count = len(q.split())
        threshold  = self._min_score
        if word_count <= _SHORT_QUERY_WORDS:
            threshold = max(0.0, self._min_score - _SHORT_QUERY_BOOST)

        q_vec = self._embed(q)
        score = float(self._anchor_vec @ q_vec)

        allowed = score >= threshold

        log.info(
            "[GeneralGuard] %s | score=%.4f | threshold=%.4f | q='%s'",
            "PASS" if allowed else "BLOCK",
            score, threshold, q[:80],
        )

        return {
            "allowed":   allowed,
            "reason":    "allowed" if allowed else "low_similarity",
            "rule":      None,
            "score":     score,
            "threshold": threshold,
        }

    def is_allowed(self, question: str) -> bool:
        """Convenience wrapper — returns True if the question should proceed."""
        return self.check(question)["allowed"]

    def status(self) -> dict:
        """Summary for the /health endpoint."""
        return {
            "min_score":         self._min_score,
            "anchor_sentences":  len(_MES_ANCHOR_SENTENCES),
            "hard_block_rules":  len(_HARD_BLOCK_RULES),
        }
