"""
Base data models and abstractions for the Hybrid Retrieval Engine.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a passage or policy chunk in the enterprise knowledge base."""
    chunk_id: str
    content: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A document chunk associated with a retrieval or rerank score."""
    chunk: DocumentChunk
    score: float
    rank: int = 1
    retrieval_method: str = "unknown"
