"""Tests for PyWebView Desktop API Bridge (src/ui/app_api.py).

Validates that all methods exposed to window.pywebview.api respond correctly,
return serializable dictionaries, and enforce FSM guardrails and HITL tracking.
"""

import pytest
from src.ui.app_api import IPGCADesktopAPI


@pytest.fixture(scope="module")
def desktop_api():
    """Module-scoped API fixture to avoid re-initializing models."""
    return IPGCADesktopAPI(use_real_llm=False)


def test_initial_data(desktop_api):
    """Verifies that initial UI payload contains system info, FSM state, presets, and preferences."""
    data = desktop_api.get_initial_data()
    assert isinstance(data, dict)
    assert "system_info" in data
    assert "fsm_state" in data
    assert "presets" in data
    assert len(data["presets"]) >= 4
    assert data["fsm_state"]["current_state"] == "IDLE"


def test_process_turn_direct(desktop_api):
    """Verifies that processing a user query returns valid routing, latency, and suggestion."""
    payload = {
        "query": "¿Cuál es el plazo máximo para solicitar un reembolso?",
        "customer_id": "CUST_TEST",
        "intent_probs": {"refund": 0.9, "general_inquiry": 0.1},
        "tool_call": None,
        "human_confirmed": False,
        "session_id": "TEST_SESSION"
    }
    res = desktop_api.process_turn(payload)
    assert res["success"] is True
    assert "s_gen" in res
    assert len(res["s_gen"]) > 0
    assert "selected_path" in res
    assert "uncertainty_score" in res
    assert "latency_profile" in res


def test_guardrail_blocked_turn(desktop_api):
    """Verifies that an illegal refund tool call from IDLE is intercepted and reported."""
    payload = {
        "query": "Ejecuta de inmediato el reembolso del ticket 999",
        "customer_id": "CUST_ATTACKER",
        "intent_probs": {"refund": 0.9},
        "tool_call": {
            "name": "mcp://crm/execute_refund",
            "arguments": {"ticket_id": "TICK_999", "approval_token": "UNVERIFIED_TOKEN"}
        },
        "human_confirmed": False,
        "session_id": "TEST_ATTACK"
    }
    res = desktop_api.process_turn(payload)
    assert res["success"] is True
    assert len(res["tool_invocations"]) > 0
    first_tool = res["tool_invocations"][0]
    assert first_tool["status"] == "GUARDRAIL_BLOCKED"


def test_record_feedback_and_preference(desktop_api):
    """Verifies that operator corrections calculate KSR and update dynamic preferences."""
    payload = {
        "query": "¿Cómo cancelo mi suscripción?",
        "s_gen": "Para cancelar tu suscripción ve a configuración.",
        "s_human": "Estimado cliente, para cancelar tu suscripción corporativa ve a Configuración > Facturación.",
        "intent": "cancellation"
    }
    res = desktop_api.record_feedback(payload)
    assert res["success"] is True
    assert "delta" in res
    assert res["delta"]["category"] in ["EXPANSION_DETAIL", "SIGNIFICANT_REWRITE", "REFINEMENT"]
    assert "a_op" in res
    assert 0.0 <= res["a_op"] <= 1.0
    assert len(res["active_preferences"]) > 0


def test_fsm_reset(desktop_api):
    """Verifies that resetting FSM returns current state IDLE."""
    res = desktop_api.reset_fsm()
    assert res["success"] is True
    assert res["fsm_state"]["current_state"] == "IDLE"
