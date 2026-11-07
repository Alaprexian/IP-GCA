"""
Hierarchical Parent-Child Document Chunker.
Implements two-level chunking strategy:
- Parent chunks: large context windows (512-1024 tokens) for coherent context injection
- Child chunks: fine-grained passages (128-256 tokens) for precise retrieval
"""

from typing import List, Tuple, Optional
import re
import yaml
from src.retrieval.base import DocumentChunk


def _split_into_sentences(text: str) -> List[str]:
    """Splits text into sentence-level units using regex."""
    # Split on period/question/exclamation followed by space or end
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _build_sliding_window(
    sentences: List[str],
    window_size: int,
    overlap: int
) -> List[str]:
    """
    Creates overlapping text windows from a list of sentences.
    Overlap ensures context continuity across adjacent chunks.
    """
    windows: List[str] = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        window = " ".join(sentences[i: i + window_size])
        if window.strip():
            windows.append(window)
    return windows


class HierarchicalChunker:
    """
    Splits documents into a two-level parent-child hierarchy:
    - Parent: Broad document sections (3-8 sentences) — used for context injection to LLM
    - Child: Fine paragraphs (1-3 sentences) — used for vector search and reranking
    """

    def __init__(
        self,
        parent_window_sentences: int = 6,
        parent_overlap_sentences: int = 2,
        child_window_sentences: int = 2,
        child_overlap_sentences: int = 1
    ):
        self.parent_window = parent_window_sentences
        self.parent_overlap = parent_overlap_sentences
        self.child_window = child_window_sentences
        self.child_overlap = child_overlap_sentences

    def chunk_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: Optional[dict] = None
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Returns (parent_chunks, child_chunks).
        Each child chunk references its parent_id in metadata for context retrieval.
        """
        metadata = metadata or {}
        sentences = _split_into_sentences(text)
        if not sentences:
            return [], []

        
        parent_windows = _build_sliding_window(
            sentences, self.parent_window, self.parent_overlap
        )
        parent_chunks: List[DocumentChunk] = []
        for i, window_text in enumerate(parent_windows):
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}__parent_{i}",
                title=title,
                content=window_text,
                metadata={
                    **metadata,
                    "level": "parent",
                    "doc_id": doc_id,
                    "window_index": i
                }
            )
            parent_chunks.append(chunk)

        child_chunks: List[DocumentChunk] = []
        for p_idx, parent_chunk in enumerate(parent_chunks):
            parent_sentences = _split_into_sentences(parent_chunk.content)
            child_windows = _build_sliding_window(
                parent_sentences, self.child_window, self.child_overlap
            )
            for c_idx, child_text in enumerate(child_windows):
                child = DocumentChunk(
                    chunk_id=f"{doc_id}__parent_{p_idx}__child_{c_idx}",
                    title=title,
                    content=child_text,
                    metadata={
                        **metadata,
                        "level": "child",
                        "doc_id": doc_id,
                        "parent_id": parent_chunk.chunk_id
                    }
                )
                child_chunks.append(child)

        return parent_chunks, child_chunks

    def chunk_multiple(
        self,
        documents: List[dict]
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Processes a list of document dicts with keys: doc_id, title, text, metadata.
        Returns aggregated (all_parents, all_children).
        """
        all_parents: List[DocumentChunk] = []
        all_children: List[DocumentChunk] = []

        for doc in documents:
            parents, children = self.chunk_document(
                doc_id=doc["doc_id"],
                title=doc.get("title", ""),
                text=doc["text"],
                metadata=doc.get("metadata", {})
            )
            all_parents.extend(parents)
            all_children.extend(children)

        return all_parents, all_children
