"""
Integrated retrieval pipeline combining HybridRetrievalEngine,
document loader, hierarchical chunker, latency tracking and GPL-Controller routing.

Demonstrates the complete Fase 2 data flow:
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
  Query → GPL-Controller routing decision → HybridRetrievalEngine → 
  RRF Fusion → Cross-Encoder Rerank → Scored context passages
"""

import json
from pathlib import Path

from src.retrieval.engine import HybridRetrievalEngine
from src.retrieval.document_loader import build_index_from_file
from src.controller.router import GPLController, ExecutionPath
from src.evaluation.latency_tracker import LatencyTracker

KB_PATH = "data/raw/corporate_knowledge_base.json"
MODEL_OVERRIDE = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_OVERRIDE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_engine(reranker: bool = True) -> HybridRetrievalEngine:
    engine = HybridRetrievalEngine(
        enable_reranker=reranker,
        dense_model_override=MODEL_OVERRIDE,
        reranker_model_override=RERANKER_OVERRIDE
    )
    stats = build_index_from_file(KB_PATH, engine)
    return engine


def run_retrieval_demo():
    print("\n" + "="*60)
    print("  IP-GCA: Retrieval Pipeline Integration Demo")
    print("="*60)

    # Build engine and index knowledge base
    print("\n[1/3] Indexing corporate knowledge base...")
    engine = build_engine(reranker=True)

    controller = GPLController("configs/config_default.yaml")
    tracker = LatencyTracker()

    # Example queries representing real customer interactions
    queries = [
        {
            "text": "¿Cuánto tiempo tengo para devolver el producto?",
            "intent_probs": {"refund_request": 0.85, "product_inquiry": 0.15}
        },
        {
            "text": "Mi laptop tiene una falla en la pantalla y quiero usar la garantía",
            "intent_probs": {"warranty_claim": 0.75, "technical_support": 0.25}
        },
        {
            "text": "Cancelar mi suscripción mensual de inmediato",
            "intent_probs": {"cancel_subscription": 0.90, "billing_inquiry": 0.10}
        },
    ]

    print("\n[2/3] Processing queries through GPL-Controller + Retrieval Engine...\n")

    for i, query_data in enumerate(queries, start=1):
        query = query_data["text"]
        intent_probs = query_data["intent_probs"]

        tracker.start_request(query)

        with tracker.measure("routing"):
            # First retrieve to get score distribution for u_ctx
            pre_results = engine.retrieve(query, top_k=5, use_reranker=False)
            retrieval_scores = engine.extract_scores(pre_results)
            decision = controller.route_query(query, retrieval_scores, intent_probs)

        # Only run reranker if budget allows
        apply_rerank = decision.selected_path in [
            ExecutionPath.HYBRID_RAG,
            ExecutionPath.AGENTIC_REFLECTION
        ]

        with tracker.measure("reranking"):
            final_results = engine.retrieve(query, top_k=3, use_reranker=apply_rerank)

        profile = tracker.finish_request(result_count=len(final_results))

        # Display results (ASCII-safe for Windows cp1252 terminal)
        path_label = {"FAST_PARAMETRIC": "[FAST]", "HYBRID_RAG": "[RAG]",
                      "AGENTIC_REFLECTION": "[AGENT]", "ASSISTED_ABSTENTION": "[ABSTAIN]"}
        label = path_label.get(decision.selected_path.value, "[?]")

        print(f"Query {i}: \"{query[:60]}...\"")
        print(f"  {label} Path: {decision.selected_path.value}  |  u_ctx={decision.uncertainty_score:.3f}  |  {profile.total_latency_ms:.0f} ms")
        print(f"  Retrieved {len(final_results)} passage(s):")
        for r in final_results:
            title = r.chunk.metadata.get("level", "") + " | " + (r.chunk.title or "?")
            snippet = r.chunk.content[:100].replace('\n', ' ')
            print(f"    [{r.rank}] score={r.score:.4f}  {title}")
            print(f"        {snippet}...")
        print()

    print("[3/3] Latency Summary:")
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n✅ Integration demo completed successfully.")


if __name__ == "__main__":
    run_retrieval_demo()
