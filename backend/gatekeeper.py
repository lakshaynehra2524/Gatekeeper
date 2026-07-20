"""
Edge Gatekeeper engine.

Loads the joblib artifacts trained in notebooks/02 and turns raw incremental
transcript fragments into gate decisions:

    FORWARD    confident that the moment is meaningful -> send downstream
    REJECT     confident that the moment is ordinary   -> drop locally
    UNCERTAIN  confidence in the grey band             -> forward, low priority
    DUPLICATE  near-repeat of a recently forwarded moment -> suppress

Everything here runs locally on CPU with no network access, matching the
edge-only constraint of the assignment.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Decision constants
FORWARD = "FORWARD"
REJECT = "REJECT"
UNCERTAIN = "UNCERTAIN"
DUPLICATE = "DUPLICATE"

FORWARD_THRESHOLD = 0.60      # p(meaningful) above this -> FORWARD
REJECT_THRESHOLD = 0.40       # p(meaningful) below this -> REJECT
DUPLICATE_SIMILARITY = 0.45   # cosine sim vs recent forwards above this -> DUPLICATE
                              # (measured: true repeats score >= 0.62, same-topic
                              #  non-repeats score <= 0.18 with this vectorizer)
DUPLICATE_MEMORY = 12         # how many recent forwarded moments to remember
CONTEXT_WINDOW = 3            # previous utterances kept as conversational context

# Cheap deterministic pre-filter: pure back-channel tokens never need ML.
_BACKCHANNEL = re.compile(
    r"^(yeah|yes|no|ok|okay|hmm+|haha+|lol|hi|hello|hey|bye|thanks|thank you|"
    r"acha|arre|cool|nice|right|sure|fine|good|great|wow|oh|true|same)"
    r"([\s,]*(yeah|ok|okay|haha+|hmm+|cool|nice|right|true|same|too|man|yaar|na))*$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text)


@dataclass
class GateResult:
    text: str
    decision: str
    label: str                 # predicted moment type (hint for downstream)
    confidence: float          # probability of the predicted label
    p_meaningful: float        # 1 - p(ordinary): drives the gate decision
    probs: dict                # full class distribution
    latency_ms: float
    reason: str
    similar_to: str | None = None
    context_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "decision": self.decision,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "p_meaningful": round(self.p_meaningful, 3),
            "probs": {k: round(v, 3) for k, v in self.probs.items()},
            "latency_ms": round(self.latency_ms, 2),
            "reason": self.reason,
            "similar_to": self.similar_to,
            "context_used": self.context_used,
        }


class GatekeeperEngine:
    """Stateful, incremental gate over a live conversation."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        self.classifier = joblib.load(models_dir / "gatekeeper_classifier.joblib")
        self.classes = list(self.classifier.classes_)
        self._ordinary_idx = self.classes.index("ordinary")
        self.reset()

    # session state
    def reset(self) -> None:
        self.context: deque[str] = deque(maxlen=CONTEXT_WINDOW)
        self.recent_forwards: deque[tuple[str, np.ndarray]] = deque(maxlen=DUPLICATE_MEMORY)
        self.stats = {FORWARD: 0, REJECT: 0, UNCERTAIN: 0, DUPLICATE: 0}

    # main entry point
    def evaluate(self, text: str) -> GateResult:
        t0 = time.perf_counter()
        norm = _normalize(text)
        ctx = list(self.context)

        # 1. deterministic back-channel filter (no ML needed)
        if not norm or _BACKCHANNEL.fullmatch(norm):
            result = GateResult(
                text=text, decision=REJECT, label="ordinary",
                confidence=1.0, p_meaningful=0.0,
                probs={c: (1.0 if c == "ordinary" else 0.0) for c in self.classes},
                latency_ms=(time.perf_counter() - t0) * 1000,
                reason="back-channel filter: pure filler, no ML run",
                context_used=ctx,
            )
            self._commit(norm, None, result)
            return result

        # 2. ML classification. The classifier scores the utterance ALONE:
        #    the vectorizer was trained on single utterances, and experiments
        #    showed that concatenating context contaminates short ordinary
        #    lines with the previous meaningful line's vocabulary. The rolling
        #    context is still tracked (reported to downstream + used by the
        #    duplicate memory); context-aware features are listed as future work.
        vec = self.vectorizer.transform([norm])
        probs = self.classifier.predict_proba(vec)[0]
        label_idx = int(np.argmax(probs))
        label = self.classes[label_idx]
        p_meaningful = float(1.0 - probs[self._ordinary_idx])
        prob_map = dict(zip(self.classes, map(float, probs)))

        # 3. duplicate suppression against recently forwarded moments
        similar_to = self._find_duplicate(vec)
        if similar_to is not None and p_meaningful >= REJECT_THRESHOLD:
            result = GateResult(
                text=text, decision=DUPLICATE, label=label,
                confidence=float(probs[label_idx]), p_meaningful=p_meaningful,
                probs=prob_map,
                latency_ms=(time.perf_counter() - t0) * 1000,
                reason=f"near-repeat of a recently forwarded moment "
                       f"(cosine ≥ {DUPLICATE_SIMILARITY})",
                similar_to=similar_to, context_used=ctx,
            )
            self._commit(norm, None, result)
            return result

        # 4. threshold gate with an explicit uncertainty band
        if p_meaningful >= FORWARD_THRESHOLD:
            decision, reason = FORWARD, (
                f"p(meaningful)={p_meaningful:.2f} ≥ {FORWARD_THRESHOLD} "
                f"-> confident forward as '{label}'")
        elif p_meaningful <= REJECT_THRESHOLD:
            decision, reason = REJECT, (
                f"p(meaningful)={p_meaningful:.2f} ≤ {REJECT_THRESHOLD} "
                f"-> confident reject (ordinary)")
        else:
            decision, reason = UNCERTAIN, (
                f"p(meaningful)={p_meaningful:.2f} inside grey band "
                f"({REJECT_THRESHOLD}–{FORWARD_THRESHOLD}) -> forwarded "
                f"with low-priority flag instead of guessing")

        result = GateResult(
            text=text, decision=decision, label=label,
            confidence=float(probs[label_idx]), p_meaningful=p_meaningful,
            probs=prob_map, latency_ms=(time.perf_counter() - t0) * 1000,
            reason=reason, context_used=ctx,
        )
        self._commit(norm, vec if decision in (FORWARD, UNCERTAIN) else None, result)
        return result

    # helpers
    def _find_duplicate(self, vec) -> str | None:
        for prev_text, prev_vec in self.recent_forwards:
            sim = float(cosine_similarity(vec, prev_vec)[0, 0])
            if sim >= DUPLICATE_SIMILARITY:
                return prev_text
        return None

    def _commit(self, norm: str, vec, result: GateResult) -> None:
        self.context.append(norm)
        if vec is not None:
            self.recent_forwards.append((result.text, vec))
        self.stats[result.decision] += 1

    # info
    def model_info(self) -> dict:
        import json
        meta_path = MODELS_DIR / "model_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        size_bytes = sum(
            p.stat().st_size for p in MODELS_DIR.glob("*.joblib")
        )
        return {
            "meta": meta,
            "model_assets_mb": round(size_bytes / 1024 / 1024, 3),
            "size_limit_mb": 25,
            "thresholds": {
                "forward": FORWARD_THRESHOLD,
                "reject": REJECT_THRESHOLD,
                "duplicate_similarity": DUPLICATE_SIMILARITY,
            },
            "context_window": CONTEXT_WINDOW,
            "stats": dict(self.stats),
        }
