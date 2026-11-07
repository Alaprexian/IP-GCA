"""
Formal Tool Invocation Precision Metric (A_tool).
Evaluates AST parameter match, type compliance, FSM state validity,
and penalizes hallucinated parameters.
"""

from typing import Dict, Any, Optional
import math
from src.guardrail.fsm import TransactionalGuardrailFSM


def calculate_tool_precision(
    tool_gen: str,
    params_gen: Dict[str, Any],
    tool_gt: str,
    params_gt: Dict[str, Any],
    fsm: Optional[TransactionalGuardrailFSM] = None,
    gamma_penalty: float = 1.0
) -> float:
    """
    Computes formal A_tool score:
    A_tool = I_tool * V_fsm * (MatchedParams / TotalGTParams) * exp(-gamma * HallucinatedRatio)
    """
    if tool_gen != tool_gt:
        return 0.0

    v_fsm = 1.0
    if fsm is not None:
        legal_triggers = fsm.get_legal_triggers()
        if tool_gen not in legal_triggers:
            v_fsm = 0.0

    if v_fsm == 0.0:
        return 0.0

    if not params_gt:
        extra_count = len(params_gen)
        return math.exp(-gamma_penalty * extra_count) if extra_count > 0 else 1.0

    matched_score = 0.0
    total_gt = len(params_gt)

    for k, v_expected in params_gt.items():
        if k in params_gen:
            v_actual = params_gen[k]
            if isinstance(v_expected, (int, float)) and isinstance(v_actual, (int, float)):
                # Tolerance check for numeric values
                if abs(v_expected - v_actual) < 1e-4:
                    matched_score += 1.0
            elif isinstance(v_expected, str) and isinstance(v_actual, str):
                if v_expected.strip().lower() == v_actual.strip().lower():
                    matched_score += 1.0
                elif v_expected.strip().lower() in v_actual.strip().lower():
                    matched_score += 0.5
            elif v_expected == v_actual:
                matched_score += 1.0

    param_recall = matched_score / total_gt

    extra_keys = set(params_gen.keys()) - set(params_gt.keys())
    hallucination_ratio = len(extra_keys) / total_gt
    penalty = math.exp(-gamma_penalty * hallucination_ratio)

    return float(v_fsm * param_recall * penalty)
