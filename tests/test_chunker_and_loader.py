"""
Unit and integration tests for the Hierarchical Chunker,
Document Loader, Latency Tracker, and integrated pipeline.
"""

import pytest
from src.retrieval.chunker import HierarchicalChunker, _split_into_sentences
from src.retrieval.document_loader import load_json_kb, build_index_from_file
from src.retrieval.engine import HybridRetrievalEngine
from src.evaluation.latency_tracker import LatencyTracker


# ─── Sentence splitting ────────────────────────────────────────────────────────

def test_sentence_splitting_basic():
    text = "Hola. ¿Cómo estás? Espero que bien."
    result = _split_into_sentences(text)
    assert len(result) >= 2


# ─── Hierarchical Chunker ──────────────────────────────────────────────────────

@pytest.fixture
def chunker():
    return HierarchicalChunker(
        parent_window_sentences=4,
        parent_overlap_sentences=1,
        child_window_sentences=2,
        child_overlap_sentences=1
    )


@pytest.fixture
def sample_doc():
    return {
        "doc_id": "test_pol_001",
        "title": "Política de Prueba",
        "text": (
            "Los clientes pueden devolver productos dentro de 30 días. "
            "El artículo debe estar sin usar y con el empaque original. "
            "Se aceptan devoluciones en tienda o por correo certificado. "
            "El reembolso se procesa en 5 a 7 días hábiles. "
            "Los artículos en liquidación no son elegibles para devolución."
        ),
        "metadata": {"category": "refund"}
    }


def test_chunker_generates_parents_and_children(chunker, sample_doc):
    parents, children = chunker.chunk_document(
        doc_id=sample_doc["doc_id"],
        title=sample_doc["title"],
        text=sample_doc["text"],
        metadata=sample_doc["metadata"]
    )
    assert len(parents) >= 1
    assert len(children) >= 1
    # Children should be shorter than parents
    assert all(
        len(c.content) <= len(p.content)
        for c in children
        for p in parents
        if c.metadata.get("parent_id") == p.chunk_id
    )


def test_child_chunks_reference_parent_id(chunker, sample_doc):
    parents, children = chunker.chunk_document(**{k: v for k, v in sample_doc.items()})
    parent_ids = {p.chunk_id for p in parents}
    for child in children:
        assert child.metadata.get("parent_id") in parent_ids


def test_chunker_multiple_documents(chunker):
    docs = [
        {"doc_id": "doc1", "title": "Doc1", "text": "Primera oración. Segunda oración. Tercera oración.", "metadata": {}},
        {"doc_id": "doc2", "title": "Doc2", "text": "Cuarta oración. Quinta oración. Sexta oración.", "metadata": {}}
    ]
    all_parents, all_children = chunker.chunk_multiple(docs)
    doc1_parents = [p for p in all_parents if p.metadata["doc_id"] == "doc1"]
    doc2_parents = [p for p in all_parents if p.metadata["doc_id"] == "doc2"]
    assert len(doc1_parents) >= 1
    assert len(doc2_parents) >= 1


# ─── Document Loader ───────────────────────────────────────────────────────────

def test_load_json_kb():
    docs = load_json_kb("data/raw/corporate_knowledge_base.json")
    assert isinstance(docs, list)
    assert len(docs) >= 5
    assert all("doc_id" in d and "text" in d for d in docs)


def test_build_index_from_file():
    engine = HybridRetrievalEngine(
        enable_reranker=False,
        dense_model_override="sentence-transformers/all-MiniLM-L6-v2"
    )
    stats = build_index_from_file("data/raw/corporate_knowledge_base.json", engine)
    assert stats["total_documents"] >= 5
    assert stats["total_child_chunks"] > stats["total_parent_chunks"]
    assert stats["indexed_chunks"] > 0

    # Verify retrieval works after indexing
    results = engine.retrieve("¿Puedo devolver algo comprado en oferta?", top_k=3)
    assert len(results) >= 1
    assert any("liquidación" in r.chunk.content.lower() or
               "devol" in r.chunk.content.lower()
               for r in results)


# ─── Latency Tracker ──────────────────────────────────────────────────────────

def test_latency_tracker_single_request():
    tracker = LatencyTracker()
    tracker.start_request("test query")
    import time
    with tracker.measure("stage_a"):
        time.sleep(0.01)
    with tracker.measure("stage_b"):
        time.sleep(0.01)
    profile = tracker.finish_request(result_count=3, ttft_ms=10.5)

    assert profile.total_latency_ms >= 20.0
    assert len(profile.stage_timings) == 2
    assert profile.result_count == 3
    assert profile.ttft_ms == 10.5


def test_latency_tracker_percentiles():
    import time
    tracker = LatencyTracker()
    for _ in range(10):
        tracker.start_request("q")
        with tracker.measure("s"):
            time.sleep(0.005)
        tracker.finish_request()

    summary = tracker.summary()
    assert summary["n_requests"] == 10
    assert summary["p95_ms"] >= summary["median_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]
