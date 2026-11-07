"""
Lightweight Context Uncertainty Estimation (u_ctx).
Computes joint retrieval dispersion and intent entropy in < 10 ms
without requiring autoregressive LLM token sampling.
"""

from typing import List, Dict, Optional
import math
import numpy as np


def compute_retrieval_margin(scores: List[float]) -> float:
    """
    Computes normalized score dispersion margin:
    M_ret = (s_top1 - s_topk) / (s_top1 + epsilon)
    Returns a score in [0.0, 1.0]. A higher margin indicates clear, confident retrieval.
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 1.0

    s1 = max(scores)
    sk = min(scores)

    if s1 <= 0:
        return 0.0

    margin = (s1 - sk) / (s1 + 1e-6)
    return float(np.clip(margin, 0.0, 1.0))


def compute_intent_entropy(intent_probabilities: Dict[str, float]) -> float:
    """
    Computes normalized Shannon entropy H_intent:
    H = - sum(p * log2(p)) / log2(|Intents|)
    Returns 0.0 for pure certainty and 1.0 for uniform ambiguity.
    """
    if not intent_probabilities:
        return 1.0

    num_classes = len(intent_probabilities)
    if num_classes <= 1:
        return 0.0

    total_prob = sum(intent_probabilities.values())
    if total_prob <= 0:
        return 1.0

    probs = [p / total_prob for p in intent_probabilities.values() if p > 0]
    raw_entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(num_classes)

    normalized_entropy = raw_entropy / max_entropy
    return float(np.clip(normalized_entropy, 0.0, 1.0))


def evaluate_joint_uncertainty(
    retrieval_scores: List[float],
    intent_probs: Dict[str, float],
    alpha: float = 0.5,
    beta: float = 0.5,
    session_id: Optional[str] = None,
    session_tracker: Optional["SessionUncertaintyTracker"] = None
) -> float:
    """
    Calculates combined context uncertainty u_ctx:
    u_ctx = alpha * (1 - M_ret) + beta * H_intent
    u_ctx in [0.0, 1.0].
    If session_id and session_tracker are provided, applies EMA session smoothing.
    """
    m_ret = compute_retrieval_margin(retrieval_scores)
    h_intent = compute_intent_entropy(intent_probs)

    u_ctx = (alpha * (1.0 - m_ret)) + (beta * h_intent)
    u_ctx = float(np.clip(u_ctx, 0.0, 1.0))

    if session_id is not None and session_tracker is not None:
        u_ctx = session_tracker.update(session_id, u_ctx)

    return u_ctx


class SessionUncertaintyTracker:
    """
    Tracks and smooths uncertainty across turns within customer sessions via EMA:
    u_ctx_smoothed(t) = ema_alpha * u_ctx_current(t) + (1 - ema_alpha) * u_ctx_session(t-1)
    Models gradual contextual convergence as multi-turn dialogue progresses.
    """

    def __init__(self, ema_alpha: float = 0.6):
        self.ema_alpha = ema_alpha
        self.sessions: Dict[str, float] = {}
        self.turn_counts: Dict[str, int] = {}

    def update(self, session_id: str, u_ctx_current: float) -> float:
        """
        Updates session history with current turn's uncertainty and returns smoothed value.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = u_ctx_current
            self.turn_counts[session_id] = 1
            return u_ctx_current

        prev_u = self.sessions[session_id]
        smoothed = (self.ema_alpha * u_ctx_current) + ((1.0 - self.ema_alpha) * prev_u)
        self.sessions[session_id] = float(np.clip(smoothed, 0.0, 1.0))
        self.turn_counts[session_id] += 1
        return self.sessions[session_id]

    def get_session_uncertainty(self, session_id: str) -> Optional[float]:
        return self.sessions.get(session_id)

    def get_turn_count(self, session_id: str) -> int:
        return self.turn_counts.get(session_id, 0)

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.turn_counts.pop(session_id, None)

    def clear(self) -> None:
        self.sessions.clear()
        self.turn_counts.clear()

