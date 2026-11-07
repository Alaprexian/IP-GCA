"""
Dense Semantic Retrieval Engine.
Uses dense representation encoders (e.g. BGE-M3) with cosine similarity
to capture abstract conceptual relations in customer queries.
"""

from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.retrieval.base import DocumentChunk, ScoredChunk


class DenseRetriever:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.encoder = SentenceTransformer(model_name_or_path, device=self.device)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_documents(self, chunks: List[DocumentChunk], batch_size: int = 32) -> None:
        """
        Computes and caches normalized dense embeddings for all document chunks.
        """
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [f"{c.title}: {c.content}" if c.title else c.content for c in chunks]
        raw_embs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.embeddings = raw_embs

    def search(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        """
        Encodes query and computes cosine similarity against cached chunk embeddings.
        """
        if self.embeddings is None or not self.chunks:
            return []

        query_emb = self.encoder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[ScoredChunk] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            results.append(
                ScoredChunk(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    retrieval_method="dense_embedding"
                )
            )

        return results
