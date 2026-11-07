"""
Deterministic Finite State Machine (FSM) Guardrail for MCP Tool Calls.
Implements the 5-tuple M_fsm = (Q, Sigma, delta, q_0, F) to isolate
non-deterministic LLM tool invocations from transactional systems.
"""

import ast
import operator as op_module
from typing import Dict, Any, List, Optional, Set, Callable
import yaml
from pydantic import BaseModel, Field


ALLOWED_COMPARATORS = {
    ast.Eq: op_module.eq,
    ast.NotEq: op_module.ne,
    ast.Lt: op_module.lt,
    ast.LtE: op_module.le,
    ast.Gt: op_module.gt,
    ast.GtE: op_module.ge,
}

ALLOWED_BINOP = {
    ast.Add: op_module.add,
    ast.Sub: op_module.sub,
    ast.Mult: op_module.mul,
    ast.Div: op_module.truediv,
}

ALLOWED_UNARYOP = {
    ast.Not: op_module.not_,
    ast.USub: op_module.neg,
}


def safe_eval_condition(condition_str: str, params: Dict[str, Any]) -> bool:
    """
    Safely evaluates boolean and comparison guard expressions using AST parsing
    without invoking arbitrary code execution (eval() free).
    Supports comparisons (==, !=, <, <=, >, >=), boolean logic (and, or, not),
    basic arithmetic (+, -, *, /), and variable lookups against `params`.
    """
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in params:
                return params[node.id]
            elif node.id in ("True", "true"):
                return True
            elif node.id in ("False", "false"):
                return False
            elif node.id in ("None", "null"):
                return None
            else:
                raise ValueError(f"Variable '{node.id}' no encontrada en parámetros.")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_UNARYOP:
                raise ValueError(f"Operador unario no permitido: {op_type.__name__}")
            return ALLOWED_UNARYOP[op_type](_eval_node(node.operand))
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_BINOP:
                raise ValueError(f"Operador binario no permitido: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return ALLOWED_BINOP[op_type](left, right)
        elif isinstance(node, ast.Compare):
            left = _eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in ALLOWED_COMPARATORS:
                    raise ValueError(f"Operador de comparación no permitido: {op_type.__name__}")
                right = _eval_node(comparator)
                if not ALLOWED_COMPARATORS[op_type](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_eval_node(val) for val in node.values)
            elif isinstance(node.op, ast.Or):
                return any(_eval_node(val) for val in node.values)
            else:
                raise ValueError(f"Operador booleano no soportado: {type(node.op).__name__}")
        else:
            raise ValueError(f"Expresión AST no permitida por seguridad: {type(node).__name__}")

    tree = ast.parse(condition_str, mode="eval")
    return bool(_eval_node(tree))


class TransitionGuardViolation(Exception):
    """Raised when an FSM transition is illegal or violates a business guard."""
    pass


class FSMTransitionResult(BaseModel):
    success: bool
    current_state: str
    previous_state: str
    trigger: str
    message: str
    requires_human_approval: bool = False
    mcp_tool: Optional[str] = None
    sanitized_params: Dict[str, Any] = Field(default_factory=dict)


class TransactionalGuardrailFSM:
    def __init__(self, config_path: str = "configs/fsm_rules.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        fsm_def = self.config["fsm_definition"]
        self.name: str = fsm_def["name"]
        self.initial_state: str = fsm_def["initial_state"]
        self.error_state: str = fsm_def["error_state"]
        self.final_states: Set[str] = set(fsm_def.get("final_states", []))

        self.current_state: str = self.initial_state
        self.state_history: List[str] = [self.initial_state]
        self.transitions: List[Dict[str, Any]] = self.config.get("transitions", [])

        # Index transitions by trigger for O(1) lookup
        self._transition_map: Dict[str, List[Dict[str, Any]]] = {}
        for t in self.transitions:
            trigger = t["trigger"]
            if trigger not in self._transition_map:
                self._transition_map[trigger] = []
            self._transition_map[trigger].append(t)

    def reset(self) -> None:
        """Reset the FSM to its initial state."""
        self.current_state = self.initial_state
        self.state_history = [self.initial_state]

    def get_legal_triggers(self) -> List[str]:
        """Return list of valid triggers from current state."""
        legal = []
        for trigger, trans_list in self._transition_map.items():
            for t in trans_list:
                src = t["source"]
                if isinstance(src, list):
                    if self.current_state in src:
                        legal.append(trigger)
                elif self.current_state == src:
                    legal.append(trigger)
        return list(set(legal))

    def evaluate_transition(
        self,
        trigger: str,
        params: Optional[Dict[str, Any]] = None,
        is_human_verified: bool = False
    ) -> FSMTransitionResult:
        """
        Validate and execute a state transition.
        Enforces schema requirements, state precedence, and business guards.
        """
        params = params or {}
        prev_state = self.current_state

        if trigger not in self._transition_map:
            return FSMTransitionResult(
                success=False,
                current_state=self.current_state,
                previous_state=prev_state,
                trigger=trigger,
                message=f"Disparador desconocido: '{trigger}'."
            )

        matched_transition = None
        for t in self._transition_map[trigger]:
            src = t["source"]
            if isinstance(src, list) and self.current_state in src:
                matched_transition = t
                break
            elif self.current_state == src:
                matched_transition = t
                break

        if not matched_transition:
            legal = self.get_legal_triggers()
            return FSMTransitionResult(
                success=False,
                current_state=self.current_state,
                previous_state=prev_state,
                trigger=trigger,
                message=(
                    f"Transición ilegal: No se puede disparar '{trigger}' desde el estado "
                    f"'{self.current_state}'. Disparadores legales actuales: {legal}"
                )
            )

        required_params = matched_transition.get("required_params", [])
        missing = [p for p in required_params if p not in params or params[p] is None]
        if missing:
            return FSMTransitionResult(
                success=False,
                current_state=self.current_state,
                previous_state=prev_state,
                trigger=trigger,
                message=f"Faltan parámetros requeridos para '{trigger}': {missing}"
            )

        guards = matched_transition.get("guards", [])
        for guard in guards:
            cond = guard.get("condition")
            if cond:
                try:
                    valid = safe_eval_condition(cond, params)
                    if not valid:
                        err_msg = guard.get("error_message", f"Violación de guarda: {cond}")
                        return FSMTransitionResult(
                            success=False,
                            current_state=self.current_state,
                            previous_state=prev_state,
                            trigger=trigger,
                            message=err_msg
                        )
                except Exception as e:
                    return FSMTransitionResult(
                        success=False,
                        current_state=self.current_state,
                        previous_state=prev_state,
                        trigger=trigger,
                        message=f"Error al evaluar guarda '{cond}': {e}"
                    )

        needs_human = matched_transition.get("is_human_gate", False)
        if needs_human and not is_human_verified:
            return FSMTransitionResult(
                success=True,
                current_state=self.current_state,
                previous_state=prev_state,
                trigger=trigger,
                message="Transición pendiente de confirmación explícita del operador humano.",
                requires_human_approval=True,
                mcp_tool=matched_transition.get("mcp_tool"),
                sanitized_params=params
            )

        dest_state = matched_transition["destination"]
        self.current_state = dest_state
        self.state_history.append(dest_state)

        return FSMTransitionResult(
            success=True,
            current_state=self.current_state,
            previous_state=prev_state,
            trigger=trigger,
            message=f"Transición exitosa: {prev_state} -> {dest_state}",
            requires_human_approval=False,
            mcp_tool=matched_transition.get("mcp_tool"),
            sanitized_params=params
        )

    def is_terminated(self) -> bool:
        """Check if FSM has reached a terminal state."""
        return self.current_state in self.final_states
