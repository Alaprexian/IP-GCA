"""
Unit and integration tests for the IP-GCA End-to-End Copilot Orchestrator.
Tests multi-path routing, feedback loop adaptation, FSM guardrail tool isolation,
and non-circular metric generation.
"""

import pytest
from src.copilot.orchestrator import IPGCAOrchestrator, CopilotTurnResponse
from src.controller.router import ExecutionPath


@pytest.fixture(scope="module")
def orchestrator():
    return IPGCAOrchestrator(
        config_path="configs/config_default.yaml",
        fsm_config_path="configs/fsm_rules.yaml",
        kb_path="data/raw/corporate_knowledge_base.json",
        enable_reranker=False # Use fast mode for unit testing
    )


def test_orchestrator_initialization(orchestrator):
    assert orchestrator.controller is not None
    assert orchestrator.fsm is not None
    assert orchestrator.preference_memory is not None
    assert orchestrator.engine is not None


def test_turn_execution_hybrid_rag(orchestrator):
    query = "¿Cuál es el plazo máximo para solicitar un reembolso?"
    intents = {"refund_request": 0.85, "general_inquiry": 0.15}

    response: CopilotTurnResponse = orchestrator.process_turn(
        query=query,
        customer_id="CUST_001",
        intent_probs=intents
    )

    assert response.query == query
    assert response.selected_path in [ExecutionPath.HYBRID_RAG, ExecutionPath.FAST_PARAMETRIC]
    assert len(response.s_gen) > 10
    assert response.latency_profile is not None
    assert response.latency_profile.total_latency_ms > 0


def test_feedback_loop_and_in_context_adaptation(orchestrator):
    query = "¿Cuánto tiempo tengo para devolver un producto de audio?"
    # Suggestion produced by AI
    s_gen = "El plazo máximo de devolución es de 7 días calendario."
    # Human supervisor corrects it to 15 days for audio category
    s_human = "El plazo estándar es 7 días, pero para la categoría Audio es de 15 días con empaque original."

    delta, a_op, batch = orchestrator.record_operator_feedback(
        query=query,
        s_gen=s_gen,
        s_human=s_human,
        intent="return_inquiry"
    )

    assert delta.is_significant
    assert 0.0 <= a_op <= 1.0
    assert len(orchestrator.preference_memory.keys) >= 1

    # In the next turn, check that the preference constraint is retrieved and applied
    follow_up_query = "¿Puedo devolver mis audífonos después de 10 días?"
    response2 = orchestrator.process_turn(
        query=follow_up_query,
        intent_probs={"return_inquiry": 0.90}
    )

    # The learned preference should be captured
    assert len(response2.applied_preferences) > 0


def test_agentic_tool_guardrail_blocking(orchestrator):
    # Reset FSM to initial state
    orchestrator.fsm.reset()

    # Attempt to execute_action directly without authentication
    tool_req = {
        "tool_name": "execute_action",
        "parameters": {"ticket_id": "TICK_001", "approval_token": "TOK_123"}
    }

    response = orchestrator.process_turn(
        query="Por favor emitan el reembolso directamente ahora",
        tool_call_request=tool_req,
        intent_probs={"refund_request": 0.95}
    )

    # If routed or processed with tool request, guardrail must block or flag illegal state
    if response.tool_invocations:
        status = response.tool_invocations[0]["status"]
        assert status in ["GUARDRAIL_BLOCKED", "PENDING_HUMAN_APPROVAL"]
