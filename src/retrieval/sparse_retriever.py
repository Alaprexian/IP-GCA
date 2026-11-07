"""
Sparse Lexical Retrieval Engine.
Implements lexical matching using BM25Okapi with vocabulary expansion hooks (SPLADE-v2).
Provides high precision for technical keywords, product codes, and policy identifiers.
"""

from typing import List, Dict, Any, Optional
import re
from rank_bm25 import BM25Okapi
from src.retrieval.base import DocumentChunk, ScoredChunk


def tokenize_text(text: str) -> List[str]:
    """Basic lexical tokenizer for search indexing."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1]


class SparseRetriever:
    def __init__(self, use_splade_expansion: bool = False):
        self.use_splade_expansion = use_splade_expansion
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Indexes a list of DocumentChunks into the sparse inverted index."""
        self.chunks = chunks
        self.corpus_tokens = [tokenize_text(f"{c.title or ''} {c.content}") for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        """
        Executes sparse lexical query and returns top_k scored chunks.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        
        
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(raw_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[ScoredChunk] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
            if score > 0.0:
                results.append(
                    ScoredChunk(
                        chunk=self.chunks[idx],
                        score=score,
                        rank=rank,
                        retrieval_method="sparse_bm25"
                    )
                )

        return results
