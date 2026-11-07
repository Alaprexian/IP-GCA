"""
Unit tests for GPL-Controller dynamic routing and uncertainty evaluation.
"""

import pytest
from src.controller.router import GPLController, ExecutionPath
from src.controller.uncertainty import (
    compute_retrieval_margin,
    compute_intent_entropy,
    evaluate_joint_uncertainty
)


@pytest.fixture
def controller():
    return GPLController("configs/config_default.yaml")


def test_retrieval_margin_calculation():
    # Clear confident retrieval: top score = 0.95, lowest = 0.10
    margin = compute_retrieval_margin([0.95, 0.50, 0.20, 0.10])
    assert margin > 0.80

    # Flat ambiguous retrieval: all scores close to 0.50
    flat_margin = compute_retrieval_margin([0.52, 0.51, 0.50, 0.49])
    assert flat_margin < 0.10


def test_intent_entropy_calculation():
    # Pure certainty: 99% probability on one intent
    entropy_certain = compute_intent_entropy({"refund_request": 0.99, "status_check": 0.01})
    assert entropy_certain < 0.15

    # High ambiguity: equal probability on 4 classes
    entropy_ambiguous = compute_intent_entropy({
        "refund_request": 0.25,
        "status_check": 0.25,
        "cancel_order": 0.25,
        "complaint": 0.25
    })
    assert entropy_ambiguous > 0.95


def test_fast_parametric_routing(controller):
    # Very high retrieval margin, very certain intent -> Low u_ctx -> FAST_PARAMETRIC
    decision = controller.route_query(
        query="Hola, ¿cuál es el horario de atención?",
        retrieval_scores=[0.98, 0.10],
        intent_probabilities={"greeting": 0.99, "other": 0.01}
    )
    assert decision.selected_path == ExecutionPath.FAST_PARAMETRIC
    assert decision.routing_latency_ms < 15.0  # Ultra-fast execution guarantee


def test_hybrid_rag_routing(controller):
    # Moderate uncertainty: intent clearly dominant (85%), retrieval margin moderate
    decision = controller.route_query(
        query="¿Cuál es la política para devolver artículos comprados en liquidación?",
        retrieval_scores=[0.80, 0.50, 0.30],
        intent_probabilities={"policy_inquiry": 0.88, "refund_request": 0.12}
    )
    assert decision.selected_path == ExecutionPath.HYBRID_RAG


def test_latency_budget_downgrade(controller):
    # Query with uncertainty in AGENTIC_REFLECTION range (0.65 <= u_ctx < 0.85)
    # but only 400 ms remaining out of 1200 ms budget
    decision = controller.route_query(
        query="Mi pedido no llegó, quiero cancelar y pedir compensación si califica.",
        retrieval_scores=[0.70, 0.50, 0.35],
        intent_probabilities={"cancel_order": 0.65, "compensation": 0.35},
        elapsed_time_ms=850.0  # Remaining budget = 350 ms, insufficient for agentic (1800 ms)
    )
    # Should downgrade to HYBRID_RAG or FAST_PARAMETRIC due to latency
    assert decision.selected_path in [ExecutionPath.HYBRID_RAG, ExecutionPath.FAST_PARAMETRIC]
    assert "Degradación por latencia" in decision.rationale


def test_session_uncertainty_accumulation(controller):
    from src.controller.uncertainty import SessionUncertaintyTracker

    tracker = SessionUncertaintyTracker(ema_alpha=0.5)
    session_id = "sess_customer_123"

    # Turn 1: High uncertainty (e.g. 0.80)
    u1 = tracker.update(session_id, 0.80)
    assert u1 == pytest.approx(0.80)
    assert tracker.get_turn_count(session_id) == 1

    # Turn 2: Query clarifies context, raw uncertainty drops to 0.40
    # Smoothed: 0.5 * 0.40 + 0.5 * 0.80 = 0.60
    u2 = tracker.update(session_id, 0.40)
    assert u2 == pytest.approx(0.60)
    assert tracker.get_turn_count(session_id) == 2

    # Turn 3: Context well understood, raw uncertainty drops to 0.20
    # Smoothed: 0.5 * 0.20 + 0.5 * 0.60 = 0.40
    u3 = tracker.update(session_id, 0.20)
    assert u3 == pytest.approx(0.40)

    # Controller route_query with session_id
    d1 = controller.route_query(
        query="Hola",
        retrieval_scores=[0.5, 0.4],
        intent_probabilities={"a": 0.5, "b": 0.5},
        session_id="sess_multi_turn_test"
    )
    assert d1.uncertainty_score > 0.0

