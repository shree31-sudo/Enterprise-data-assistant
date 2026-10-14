from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("QuestionValidator")

# ── project root: one level above core/ ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── config from .env ──────────────────────────────────────────────────────────
_THRESHOLD: float = float(
    os.getenv("VALIDATOR_THRESHOLD", "0.50").strip()
)

_CLASSIFIER_PATH: Path = Path(
    os.getenv(
        "VALIDATOR_CLASSIFIER_PATH",
        str(_PROJECT_ROOT / "data" / "scope_classifier.pkl"),
    ).strip()
)

_REJECT_MESSAGE: str = os.getenv(
    "VALIDATOR_REJECT_MESSAGE",
    (
        "I can only answer questions about the MES application "
        "(production orders, BOMs, inventory, work centres, quality, "
        "scheduling, and related manufacturing topics). "
        "Please rephrase your question in that context."
    ),
).strip()


def _safe_vec(vec: np.ndarray) -> np.ndarray:
    """
    FIX #2 / #6 — Normalise embedding shape to 1-D before classifier use.

    SentenceTransformer.encode() for a single string can return either
    shape (D,) or (1, D) depending on the caller.  predict_proba expects
    a 2-D input [vec], so the inner vec must be 1-D.  Squeezing here
    prevents a silent fail-open caused by a shape mismatch.
    """
    v = np.asarray(vec)
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]          # (1, D) → (D,)
    elif v.ndim != 1:
        raise ValueError(
            f"Unexpected embedding shape {v.shape}; expected 1-D vector."
        )
    return v


class QuestionValidator:
    """
    Validates whether a question is within the MES/MFC application scope
    using a pre-trained logistic regression classifier.

    Parameters
    ----------
    embedder
        The SentenceTransformer instance already loaded in the pipeline.
        Reused here — no second model is loaded.
    threshold : float
        Minimum in-scope probability required to accept a question.
        Defaults to VALIDATOR_THRESHOLD env var (0.50).
    classifier_path : Path
        Path to scope_classifier.pkl from train_scope_classifier.py.
        Defaults to VALIDATOR_CLASSIFIER_PATH env var.
    """

    def __init__(
        self,
        embedder,
        threshold: float      = _THRESHOLD,
        classifier_path: Path = _CLASSIFIER_PATH,
    ) -> None:
        self._embedder  = embedder
        self._threshold = threshold
        self._clf       = self._load_classifier(classifier_path)
        self._clf_path  = classifier_path   # kept for reload_classifier()

        if self._clf is not None:
            log.info(
                f"[QuestionValidator] ready — "
                f"threshold={self._threshold}  model={classifier_path.name}"
            )
        else:
            log.warning(
                "[QuestionValidator] classifier not loaded — "
                "all questions will pass through (fail-open mode). "
                "Run train_scope_classifier.py to enable scope validation."
            )

    # -------------------------------------------------------------------------
    # Classifier I/O
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_classifier(path: Path):
        """
        Load the sklearn Pipeline from disk.
        Returns None (fail-open) if the file is missing or corrupt.
        """
        if not path.exists():
            log.warning(
                f"[QuestionValidator] Classifier file not found: {path}. "
                "Scope validation is DISABLED. "
                "Run train_scope_classifier.py to generate it."
            )
            return None
        try:
            with open(path, "rb") as fh:
                clf = pickle.load(fh)
            log.info(f"[QuestionValidator] Classifier loaded from {path}")
            return clf
        except Exception as exc:
            log.error(
                f"[QuestionValidator] Failed to load classifier ({exc}). "
                "Scope validation is DISABLED."
            )
            return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def validate(
        self,
        question: str,
        precomputed_vec: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether *question* is within scope.

        Parameters
        ----------
        question : str
            Raw user question string.
        precomputed_vec : np.ndarray, optional
            Normalised embedding of the question.  Pass this when the
            caller has already embedded the question (e.g. SemanticCache)
            to avoid a redundant encode() call.  Both (D,) and (1, D)
            shapes are accepted — normalised internally.

        Returns
        -------
        (True,  "")       — accepted: proceed with the pipeline.
        (False, message)  — rejected: return message directly to the user.
        """
        # Classifier absent → fail open, never block the pipeline
        if self._clf is None:
            return True, ""

        try:
            if precomputed_vec is not None:
                # FIX #2 — guard against (1, D) shape from encode()
                vec = _safe_vec(precomputed_vec)
            else:
                vec = self._embedder.encode(
                    question.strip(), normalize_embeddings=True
                )
                vec = _safe_vec(vec)

            # predict_proba shape: (1, 2) → index 1 = P(in-scope)
            prob_in_scope: float = float(
                self._clf.predict_proba([vec])[0][1]
            )

            log.info(
                f"[QuestionValidator] score={prob_in_scope:.4f}  "
                f"threshold={self._threshold}  q={question[:80]}"
            )

            if prob_in_scope >= self._threshold:
                return True, ""

            log.info(
                f"[QuestionValidator] REJECTED score={prob_in_scope:.4f}  "
                f"q={question[:80]}"
            )
            return False, _REJECT_MESSAGE

        except Exception as exc:
            # Any runtime failure → fail open so legitimate queries never break
            log.warning(
                f"[QuestionValidator] Runtime error ({exc}) — failing open."
            )
            return True, ""

    def score(self, question: str) -> float:
        """
        Return the raw in-scope probability for *question*.

        Useful for threshold calibration and debugging:
            validator.score("show open work orders today")       # → ~0.97
            validator.score("what is the weather in Chennai")    # → ~0.02
        """
        if self._clf is None:
            return 1.0
        try:
            # FIX #6 — same shape guard as validate() so score() never raises
            vec = self._embedder.encode(question.strip(), normalize_embeddings=True)
            vec = _safe_vec(vec)
            return float(self._clf.predict_proba([vec])[0][1])
        except Exception as exc:
            log.warning(
                f"[QuestionValidator] score() runtime error ({exc}) — returning 1.0."
            )
            return 1.0

    def update_threshold(self, new_threshold: float) -> None:
        """
        Hot-update the acceptance threshold at runtime without restarting.

        Example — tighten scope temporarily during testing:
            pipeline._validator.update_threshold(0.65)
        """
        old = self._threshold
        self._threshold = new_threshold
        log.info(f"[QuestionValidator] threshold updated: {old} → {new_threshold}")

    def reload_classifier(self, path: Optional[Path] = None) -> bool:
        """
        Hot-reload the classifier from disk after retraining.
        Returns True on success, False on failure.

        Call from the /admin/reload-validator endpoint instead of
        restarting the whole server:

            @app.post("/admin/reload-validator", dependencies=[Depends(_verify_api_key)])
            def reload_validator():
                ok = pipeline._validator.reload_classifier()
                return {"status": "reloaded" if ok else "failed"}
        """
        target = path or self._clf_path
        clf    = self._load_classifier(target)
        if clf is None:
            return False
        self._clf      = clf
        self._clf_path = target
        log.info(f"[QuestionValidator] Classifier hot-reloaded from {target}")
        return True
