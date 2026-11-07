"""
Unit and integration tests for Hybrid Retrieval Engine.
Validates sparse BM25, dense embeddings, RRF fusion, and two-stage retrieval.
"""

import pytest
from src.retrieval.base import DocumentChunk
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion, weighted_relative_score_fusion
from src.retrieval.engine import HybridRetrievalEngine


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            title="Política de Devolución Estándar",
            content="Los clientes pueden solicitar un reembolso completo dentro de los primeros 30 días posteriores a la compra con recibo original. Código de política: POL-30D.",
            metadata={"category": "refund", "department": "billing"}
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            title="Garantía de Dispositivos Electrónicos",
            content="Los artículos electrónicos cuentan con una garantía limitada del fabricante por 12 meses contra defectos de fábrica. No cubre daños por agua.",
            metadata={"category": "warranty", "department": "technical"}
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            title="Cancelación de Suscripciones Digitales",
            content="Las suscripciones recurrentes pueden cancelarse en cualquier momento desde el panel del usuario. El acceso continuará hasta el final del ciclo de facturación.",
            metadata={"category": "subscription", "department": "billing"}
        )
    ]


def test_sparse_retrieval_exact_code(sample_chunks):
    retriever = SparseRetriever()
    retriever.index_documents(sample_chunks)

    # Searching exact policy code "POL-30D"
    results = retriever.search("POL-30D", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "chunk_1"


def test_dense_retrieval_semantic(sample_chunks):
    # Use lightweight model for fast testing
    retriever = DenseRetriever(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    retriever.index_documents(sample_chunks)

    # Semantic query without exact keyword matches
    results = retriever.search("¿Cuánto tiempo tengo para que me devuelvan mi dinero?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_1"
    assert results[0].score > 0.0


def test_rrf_fusion(sample_chunks):
    retriever_sparse = SparseRetriever()
    retriever_sparse.index_documents(sample_chunks)
    sparse_res = retriever_sparse.search("garantía defectos", top_k=2)

    retriever_dense = DenseRetriever(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    retriever_dense.index_documents(sample_chunks)
    dense_res = retriever_dense.search("problemas técnicos del producto", top_k=2)

    fused = reciprocal_rank_fusion(dense_results=dense_res, sparse_results=sparse_res, top_k=2)
    assert len(fused) > 0
    # chunk_2 should be prioritized in both
    assert fused[0].chunk.chunk_id == "chunk_2"


def test_hybrid_engine_end_to_end(sample_chunks):
    # Initialize engine without cross-encoder for fast integration test
    engine = HybridRetrievalEngine(
        enable_reranker=False,
        dense_model_override="sentence-transformers/all-MiniLM-L6-v2"
    )
    engine.index_chunks(sample_chunks)

    results = engine.retrieve("Quiero cancelar mi plan mensual", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "chunk_3"

    # Verify score extraction for GPL-Controller
    scores = engine.extract_scores(results)
    assert len(scores) == len(results)
    assert all(isinstance(s, float) for s in scores)
