"""
Hybrid Search Fusion Algorithms.
Implements Reciprocal Rank Fusion (RRF) and Weighted Relative Score Fusion
to merge dense and sparse retrieval result sets.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from src.retrieval.base import ScoredChunk, DocumentChunk


def reciprocal_rank_fusion(
    dense_results: List[ScoredChunk],
    sparse_results: List[ScoredChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10
) -> List[ScoredChunk]:
    """
    Combines dense and sparse results using Reciprocal Rank Fusion (RRF):
    RRF(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, DocumentChunk] = {}

    for scored in dense_results:
        cid = scored.chunk.chunk_id
        chunk_map[cid] = scored.chunk
        rrf_scores[cid] += dense_weight / (k + scored.rank)

    for scored in sparse_results:
        cid = scored.chunk.chunk_id
        chunk_map[cid] = scored.chunk
        rrf_scores[cid] += sparse_weight / (k + scored.rank)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_results: List[ScoredChunk] = []
    for rank, (cid, score) in enumerate(sorted_chunks[:top_k], start=1):
        fused_results.append(
            ScoredChunk(
                chunk=chunk_map[cid],
                score=float(score),
                rank=rank,
                retrieval_method="hybrid_rrf"
            )
        )

    return fused_results


def weighted_relative_score_fusion(
    dense_results: List[ScoredChunk],
    sparse_results: List[ScoredChunk],
    alpha: float = 0.65,
    top_k: int = 10
) -> List[ScoredChunk]:
    """
    Combines dense and sparse results via min-max normalized weighted linear fusion:
    Score(d) = alpha * Norm(Score_dense) + (1 - alpha) * Norm(Score_sparse)
    """
    chunk_map: Dict[str, DocumentChunk] = {}
    dense_map: Dict[str, float] = {}
    sparse_map: Dict[str, float] = {}

    if dense_results:
        d_scores = [r.score for r in dense_results]
        d_min, d_max = min(d_scores), max(d_scores)
        d_range = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
        for r in dense_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
            dense_map[r.chunk.chunk_id] = (r.score - d_min) / d_range

    if sparse_results:
        s_scores = [r.score for r in sparse_results]
        s_min, s_max = min(s_scores), max(s_scores)
        s_range = (s_max - s_min) if (s_max - s_min) > 1e-6 else 1.0
        for r in sparse_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
            sparse_map[r.chunk.chunk_id] = (r.score - s_min) / s_range

    all_ids = set(dense_map.keys()).union(set(sparse_map.keys()))
    combined: List[Tuple[str, float]] = []

    for cid in all_ids:
        s_dense = dense_map.get(cid, 0.0)
        s_sparse = sparse_map.get(cid, 0.0)
        final_score = (alpha * s_dense) + ((1.0 - alpha) * s_sparse)
        combined.append((cid, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)

    results: List[ScoredChunk] = []
    for rank, (cid, score) in enumerate(combined[:top_k], start=1):
        results.append(
            ScoredChunk(
                chunk=chunk_map[cid],
                score=float(score),
                rank=rank,
                retrieval_method="hybrid_weighted_score"
            )
        )

    return results
