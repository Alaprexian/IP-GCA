"""
Unified Hybrid Retrieval Engine (SPLADE/BM25 + BGE-M3 + Cross-Encoder Reranker).
Provides single-entry API for knowledge base indexing and two-stage hybrid retrieval.
"""

from typing import List, Optional
import yaml
from src.retrieval.base import DocumentChunk, ScoredChunk
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion, weighted_relative_score_fusion
from src.retrieval.reranker import CrossEncoderReranker


class HybridRetrievalEngine:
    def __init__(
        self,
        config_path: str = "configs/config_default.yaml",
        enable_reranker: bool = True,
        dense_model_override: Optional[str] = None,
        reranker_model_override: Optional[str] = None
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        models_cfg = cfg.get("models", {})
        retrieval_cfg = cfg.get("retrieval", {})

        dense_model = dense_model_override or models_cfg.get("dense_retriever", "BAAI/bge-small-en-v1.5")
        reranker_model = reranker_model_override or models_cfg.get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.top_k_retrieval = retrieval_cfg.get("top_k_retrieval", 10)
        self.top_k_rerank = retrieval_cfg.get("top_k_rerank", 3)
        self.rrf_k = retrieval_cfg.get("rrf_k", 60)
        self.dense_weight = float(retrieval_cfg.get("dense_weight", 0.65))
        self.sparse_weight = float(retrieval_cfg.get("sparse_weight", 0.35))

        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(model_name_or_path=dense_model)

        self.enable_reranker = enable_reranker
        self.reranker = CrossEncoderReranker(model_name_or_path=reranker_model) if enable_reranker else None

        self.indexed_chunks: List[DocumentChunk] = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes document chunks into both sparse and dense retrieval systems."""
        self.indexed_chunks = chunks
        self.sparse_retriever.index_documents(chunks)
        self.dense_retriever.index_documents(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        fusion_method: str = "rrf"
    ) -> List[ScoredChunk]:
        """
        Executes two-stage hybrid retrieval:
        Stage 1: Primary hybrid retrieval (Dense + Sparse) fused via RRF or Weighted Score.
        Stage 2: Precision Cross-Encoder reranking (if enabled).
        """
        k_retrieve = self.top_k_retrieval
        k_final = top_k or self.top_k_rerank
        apply_rerank = self.enable_reranker if use_reranker is None else use_reranker

        sparse_candidates = self.sparse_retriever.search(query, top_k=k_retrieve)
        dense_candidates = self.dense_retriever.search(query, top_k=k_retrieve)

        if fusion_method == "rrf":
            fused_candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                k=self.rrf_k,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                top_k=k_retrieve
            )
        else:
            fused_candidates = weighted_relative_score_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                alpha=self.dense_weight,
                top_k=k_retrieve
            )

        if not fused_candidates:
            return []

        if apply_rerank and self.reranker:
            final_results = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_k=k_final
            )
        else:
            final_results = fused_candidates[:k_final]

        return final_results

    @staticmethod
    def extract_scores(results: List[ScoredChunk]) -> List[float]:
        """Utility to extract score distribution for GPL-Controller uncertainty evaluation."""
        return [r.score for r in results]
