"""
Unit tests for Formal Metrics: A_op, KSR, DAR, and A_tool.
"""

import pytest
from src.evaluation.alignment import (
    calculate_ksr,
    calculate_operator_alignment,
    calculate_dar
)
from src.evaluation.tool_precision import calculate_tool_precision
from src.guardrail.fsm import TransactionalGuardrailFSM


def test_ksr_exact_match():
    text = "Estimado cliente, su reembolso de 50 USD ha sido procesado."
    assert calculate_ksr(text, text) == 1.0


def test_ksr_complete_rewrite():
    s_gen = "Hola"
    s_human = "Estimado cliente, le informamos sobre su pedido."
    ksr = calculate_ksr(s_gen, s_human)
    assert ksr < 0.2


def test_operator_alignment_composite():
    s_gen = "Estimado cliente, su caso fue registrado con el ticket 102."
    s_human = "Estimado cliente, su caso ha sido registrado con el ticket 102."
    a_op = calculate_operator_alignment(s_gen, s_human)
    assert 0.80 <= a_op <= 1.0


def test_dar_calculation():
    scores = [1.0, 0.95, 0.92, 0.60, 0.40]
    dar = calculate_dar(scores, threshold=0.90)
    assert dar == 3 / 5


def test_tool_precision_exact():
    tool_gt = "authenticate_customer"
    params_gt = {"customer_id": "CUST_42"}

    tool_gen = "authenticate_customer"
    params_gen = {"customer_id": "CUST_42"}

    score = calculate_tool_precision(tool_gen, params_gen, tool_gt, params_gt)
    assert score == 1.0


def test_tool_precision_hallucinated_params():
    tool_gt = "authenticate_customer"
    params_gt = {"customer_id": "CUST_42"}

    tool_gen = "authenticate_customer"
    # Includes extra hallucinated parameter
    params_gen = {"customer_id": "CUST_42", "credit_card": "411111111"}

    score = calculate_tool_precision(tool_gen, params_gen, tool_gt, params_gt)
    assert score < 1.0  # Penalized by exponential penalty factor


def test_tool_precision_fsm_invalid_state():
    fsm = TransactionalGuardrailFSM("configs/fsm_rules.yaml")
    # In IDLE state, execute_action is NOT legal
    score = calculate_tool_precision(
        tool_gen="execute_action",
        params_gen={"ticket_id": "T1", "approval_token": "TOK"},
        tool_gt="execute_action",
        params_gt={"ticket_id": "T1", "approval_token": "TOK"},
        fsm=fsm
    )
    assert score == 0.0  # Rejected by FSM precedence


def test_faithfulness_grounded_claims():
    from src.evaluation.faithfulness import FaithfulnessEvaluator
    evaluator = FaithfulnessEvaluator(load_model=False)

    context = ["La garantía cubre defectos de fábrica durante 12 meses. El cliente debe presentar comprobante de pago."]
    s_gen = "La garantía cubre defectos de fábrica durante 12 meses. Además, el cliente debe presentar el comprobante de pago."

    result = evaluator.evaluate(s_gen, context)
    assert result.faithfulness_score >= 0.90
    assert result.hallucination_rate <= 0.10
    assert result.total_claims == 2
    assert result.supported_claims == 2


def test_faithfulness_hallucinated_claim():
    from src.evaluation.faithfulness import FaithfulnessEvaluator
    evaluator = FaithfulnessEvaluator(load_model=False)

    context = ["El horario de atención es de lunes a viernes de 9am a 6pm."]
    # The second claim about refunds is completely missing from context
    s_gen = "El horario de atención es de lunes a viernes de 9am a 6pm. Entregamos reembolsos en criptomonedas de inmediato."

    result = evaluator.evaluate(s_gen, context)
    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.faithfulness_score == 0.50
    assert result.hallucination_rate == 0.50
