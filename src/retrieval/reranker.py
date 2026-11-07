"""
Cross-Encoder Reranking Module.
Performs full cross-attention over query-document pairs to produce calibrated relevance scores.
"""

from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder
from src.retrieval.base import ScoredChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self._reranker: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.model_name, device=self.device)
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 3
    ) -> List[ScoredChunk]:
        """
        Reranks a list of candidate ScoredChunks and returns the top_k refined results.
        """
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, f"{c.chunk.title or ''} {c.chunk.content}") for c in candidates]

        scores = model.predict(pairs, show_progress_bar=False)

        scored_candidates = []
        for orig_candidate, score in zip(candidates, scores):
            scored_candidates.append(
                ScoredChunk(
                    chunk=orig_candidate.chunk,
                    score=float(score),
                    rank=1,
                    retrieval_method="cross_encoder_rerank"
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        for rank, item in enumerate(scored_candidates[:top_k], start=1):
            item.rank = rank

        return scored_candidates[:top_k]
