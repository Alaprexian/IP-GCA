"""
Formal Operator Alignment Metrics (A_op, KSR, Semantic Preservation, DAR).
Implements the multi-dimensional alignment index between AI suggestions and human edits.
"""

from typing import List, Optional, Union
import numpy as np
import Levenshtein


def calculate_ksr(s_gen: str, s_human: str) -> float:
    """
    Keystroke Savings Rate (KSR).
    Measures the mechanical reduction in editing effort:
    KSR = 1 - (LevenshteinDistance(S_gen, S_human) / max(|S_gen|, |S_human|))
    Returns 1.0 for perfect acceptance without edits.
    """
    if not s_gen and not s_human:
        return 1.0
    max_len = max(len(s_gen), len(s_human))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(s_gen, s_human)
    return 1.0 - (dist / max_len)


def calculate_semantic_similarity(
    s_gen: str,
    s_human: str,
    embedder: Optional[any] = None
) -> float:
    """
    Directional Semantic Preservation (Sim_sem).
    Computes cosine similarity between contextual embeddings of S_gen and S_human.
    If no embedder is provided, falls back to rapid lexical token Jaccard similarity.
    """
    if s_gen.strip() == s_human.strip():
        return 1.0

    if embedder is not None:
        embeddings = embedder.encode([s_gen, s_human])
        vec1 = embeddings[0] / np.linalg.norm(embeddings[0])
        vec2 = embeddings[1] / np.linalg.norm(embeddings[1])
        return float(np.dot(vec1, vec2))

    # Fast fallback: Token-level Jaccard similarity
    tokens_gen = set(s_gen.lower().split())
    tokens_human = set(s_human.lower().split())
    union = tokens_gen.union(tokens_human)
    if not union:
        return 1.0
    return len(tokens_gen.intersection(tokens_human)) / len(union)


def calculate_operator_alignment(
    s_gen: str,
    s_human: str,
    w_ksr: float = 0.40,
    w_sem: float = 0.60,
    embedder: Optional[any] = None
) -> float:
    """
    Composite Operator Alignment Index (A_op):
    A_op = w_ksr * clip(KSR, 0, 1) + w_sem * Sim_sem
    """
    ksr = max(0.0, calculate_ksr(s_gen, s_human))
    sim_sem = max(0.0, calculate_semantic_similarity(s_gen, s_human, embedder=embedder))
    return float((w_ksr * ksr) + (w_sem * sim_sem))


def calculate_dar(ksr_history: List[float], threshold: float = 0.90) -> float:
    """
    Direct Acceptance Rate (DAR).
    Proportion of suggestions accepted with minimal or zero operator edits.
    """
    if not ksr_history:
        return 0.0
    accepted = sum(1 for score in ksr_history if score >= threshold)
    return accepted / len(ksr_history)
