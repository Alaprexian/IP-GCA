"""
Unit tests for TransactionalGuardrailFSM.
Verifies state precedence, parameter guards, and human approval isolation.
"""

import pytest
from src.guardrail.fsm import TransactionalGuardrailFSM


@pytest.fixture
def fsm():
    return TransactionalGuardrailFSM("configs/fsm_rules.yaml")


def test_initial_state(fsm):
    assert fsm.current_state == "IDLE"
    assert "authenticate_customer" in fsm.get_legal_triggers()


def test_illegal_transition_rejection(fsm):
    # Attempting to execute refund directly from IDLE should be rejected
    res = fsm.evaluate_transition("execute_action", {"ticket_id": "T123", "approval_token": "TOK1"})
    assert not res.success
    assert fsm.current_state == "IDLE"
    assert "Transición ilegal" in res.message


def test_valid_sequential_flow(fsm):
    # 1. Authenticate
    r1 = fsm.evaluate_transition("authenticate_customer", {"customer_id": "CUST_99"})
    assert r1.success
    assert fsm.current_state == "AUTHENTICATED"

    # 2. Fetch ticket
    r2 = fsm.evaluate_transition("fetch_ticket", {"ticket_id": "TCK_404"})
    assert r2.success
    assert fsm.current_state == "TICKET_LOADED"

    # 3. Evaluate policy
    r3 = fsm.evaluate_transition("evaluate_policy", {"policy_category": "electronics_refund"})
    assert r3.success
    assert fsm.current_state == "POLICY_EVALUATED"

    # 4. Prepare action within business limits
    r4 = fsm.evaluate_transition("prepare_action", {"action_type": "refund", "amount": 120.0})
    assert r4.success
    assert fsm.current_state == "ACTION_PREPARED"

    # 5. Human operator gate
    r5 = fsm.evaluate_transition("human_operator_approve", is_human_verified=True)
    assert r5.success
    assert fsm.current_state == "OPERATOR_APPROVED"

    # 6. Execute mutation
    r6 = fsm.evaluate_transition("execute_action", {"ticket_id": "TCK_404", "approval_token": "SECRET"})
    assert r6.success
    assert fsm.current_state == "MUTATION_EXECUTED"


def test_business_guard_rejection(fsm):
    # Advance to POLICY_EVALUATED
    fsm.evaluate_transition("authenticate_customer", {"customer_id": "CUST_1"})
    fsm.evaluate_transition("fetch_ticket", {"ticket_id": "TCK_1"})
    fsm.evaluate_transition("evaluate_policy", {"policy_category": "return"})

    # Attempt amount > 500 (guard is amount <= 500)
    res = fsm.evaluate_transition("prepare_action", {"action_type": "refund", "amount": 950.0})
    assert not res.success
    assert "Monto excede el límite permitido" in res.message
    assert fsm.current_state == "POLICY_EVALUATED"


def test_safe_eval_condition_security():
    from src.guardrail.fsm import safe_eval_condition

    # Standard valid conditions
    assert safe_eval_condition("amount <= 500", {"amount": 300}) is True
    assert safe_eval_condition("amount <= 500", {"amount": 750}) is False
    assert safe_eval_condition("token != 'UNVERIFIED'", {"token": "VERIFIED_123"}) is True
    assert safe_eval_condition("token != 'UNVERIFIED'", {"token": "UNVERIFIED"}) is False
    assert safe_eval_condition("amount > 100 and status == 'APPROVED'", {"amount": 200, "status": "APPROVED"}) is True

    # Security: arbitrary function calls or code injection must be blocked
    with pytest.raises(ValueError, match="Expresión AST no permitida por seguridad"):
        safe_eval_condition("__import__('os').system('echo pwned')", {})

    with pytest.raises(ValueError, match="Expresión AST no permitida por seguridad"):
        safe_eval_condition("open('/etc/passwd').read()", {})

