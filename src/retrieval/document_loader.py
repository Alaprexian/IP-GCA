"""
Document Loader and Knowledge Base Indexer.
Loads raw JSON documents, applies HierarchicalChunker, and builds the HybridRetrievalEngine index.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from src.retrieval.base import DocumentChunk
from src.retrieval.chunker import HierarchicalChunker
from src.retrieval.engine import HybridRetrievalEngine


def load_json_kb(kb_path: str) -> List[dict]:
    """Loads a list of document dicts from a JSON knowledge base file."""
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index_from_file(
    kb_path: str,
    engine: HybridRetrievalEngine,
    chunker: Optional[HierarchicalChunker] = None,
    use_child_chunks: bool = True
) -> dict:
    """
    Complete pipeline:
    1. Load raw documents from JSON
    2. Apply hierarchical chunking (parent-child)
    3. Index child chunks into the HybridRetrievalEngine

    Returns a dict with indexing summary statistics.
    """
    if chunker is None:
        chunker = HierarchicalChunker(
            parent_window_sentences=5,
            parent_overlap_sentences=1,
            child_window_sentences=2,
            child_overlap_sentences=1
        )

    raw_docs = load_json_kb(kb_path)
    all_parents, all_children = chunker.chunk_multiple(raw_docs)

    # Index child chunks for precise retrieval, keep parents for context injection
    chunks_to_index = all_children if use_child_chunks else all_parents
    engine.index_chunks(chunks_to_index)

    return {
        "total_documents": len(raw_docs),
        "total_parent_chunks": len(all_parents),
        "total_child_chunks": len(all_children),
        "indexed_chunks": len(chunks_to_index),
        "kb_file": kb_path
    }


def load_parent_map(kb_path: str) -> dict:
    """
    Returns a mapping {parent_id: parent_content} for context retrieval.
    Used when we need to expand a matched child chunk to its full parent context.
    """
    chunker = HierarchicalChunker()
    raw_docs = load_json_kb(kb_path)
    all_parents, _ = chunker.chunk_multiple(raw_docs)
    return {p.chunk_id: p for p in all_parents}
