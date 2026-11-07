"""
Gated Preference-Latency Controller (GPL-Controller).
Dynamically routes customer queries along variable-depth execution paths
based on the joint uncertainty vector (u_ctx) and the remaining latency budget (l_budget).
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import time
import yaml
from pydantic import BaseModel
from src.controller.uncertainty import evaluate_joint_uncertainty, SessionUncertaintyTracker


class ExecutionPath(str, Enum):
    FAST_PARAMETRIC = "FAST_PARAMETRIC"       # Inferencia directa (< 350 ms)
    HYBRID_RAG = "HYBRID_RAG"                 # Búsqueda SPLADE + BGE-M3 + Rerank (< 1200 ms)
    AGENTIC_REFLECTION = "AGENTIC_REFLECTION" # ReAct multi-hop con llamadas a herramientas (< 2200 ms)
    ASSISTED_ABSTENTION = "ASSISTED_ABSTENTION" # Derivación a operador humano con resumen contextual


class RoutingDecision(BaseModel):
    selected_path: ExecutionPath
    uncertainty_score: float
    remaining_latency_budget_ms: float
    routing_latency_ms: float
    rationale: str


class GPLController:
    def __init__(self, config_path: str = "configs/config_default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        router_cfg = cfg.get("router", {})
        self.single_step_budget_ms = float(router_cfg.get("single_step_latency_budget_ms", 1200))
        self.multi_hop_budget_ms = float(router_cfg.get("multi_hop_latency_budget_ms", 2200))

        self.tau_fast = float(router_cfg.get("tau_fast_parametric", 0.25))
        self.tau_hybrid = float(router_cfg.get("tau_hybrid_rag", 0.65))
        self.tau_agent = float(router_cfg.get("tau_agent_reflection", 0.85))

        # Cost estimates per execution branch in milliseconds
        self.cost_estimates = {
            ExecutionPath.FAST_PARAMETRIC: 250.0,
            ExecutionPath.HYBRID_RAG: 800.0,
            ExecutionPath.AGENTIC_REFLECTION: 1800.0,
            ExecutionPath.ASSISTED_ABSTENTION: 50.0,
        }

        self.session_tracker = SessionUncertaintyTracker()

    def route_query(
        self,
        query: str,
        retrieval_scores: List[float],
        intent_probabilities: Dict[str, float],
        elapsed_time_ms: float = 0.0,
        is_multi_hop_eligible: bool = True,
        session_id: Optional[str] = None
    ) -> RoutingDecision:
        """
        Determines the optimal execution path under joint (u_ctx, l_budget) constraints.
        Executes in < 5 ms.
        """
        t0 = time.perf_counter()

        max_budget = self.multi_hop_budget_ms if is_multi_hop_eligible else self.single_step_budget_ms
        remaining_budget = max(0.0, max_budget - elapsed_time_ms)

        u_ctx = evaluate_joint_uncertainty(
            retrieval_scores,
            intent_probabilities,
            session_id=session_id,
            session_tracker=self.session_tracker
        )

        if u_ctx < self.tau_fast:
            candidate_path = ExecutionPath.FAST_PARAMETRIC
            reason = f"Baja incertidumbre (u_ctx={u_ctx:.3f} < {self.tau_fast}). Camino paramétrico suficiente."
        elif u_ctx < self.tau_hybrid:
            candidate_path = ExecutionPath.HYBRID_RAG
            reason = f"Incertidumbre moderada (u_ctx={u_ctx:.3f}). Requiere contexto RAG híbrido y reranking."
        elif u_ctx < self.tau_agent and is_multi_hop_eligible:
            candidate_path = ExecutionPath.AGENTIC_REFLECTION
            reason = f"Incertidumbre alta/compleja (u_ctx={u_ctx:.3f}). Requiere bucle agéntico multi-paso."
        else:
            candidate_path = ExecutionPath.ASSISTED_ABSTENTION
            reason = f"Incertidumbre extrema o fuera de alcance (u_ctx={u_ctx:.3f}). Se activa abstención asistida."

        selected_path = candidate_path
        if remaining_budget < self.cost_estimates[candidate_path]:
            if candidate_path == ExecutionPath.AGENTIC_REFLECTION and remaining_budget >= self.cost_estimates[ExecutionPath.HYBRID_RAG]:
                selected_path = ExecutionPath.HYBRID_RAG
                reason += f" [Degradación por latencia: presupuesto restante ({remaining_budget:.1f} ms) insuficiente para agente, ejecutando RAG]."
            elif remaining_budget >= self.cost_estimates[ExecutionPath.FAST_PARAMETRIC]:
                selected_path = ExecutionPath.FAST_PARAMETRIC
                reason += f" [Degradación por latencia: ejecutando atajo paramétrico para respetar presupuesto]."
            else:
                selected_path = ExecutionPath.ASSISTED_ABSTENTION
                reason += f" [Presupuesto de latencia agotado ({remaining_budget:.1f} ms). Derivando inmediatamente a operador]."

        routing_time_ms = (time.perf_counter() - t0) * 1000.0

        return RoutingDecision(
            selected_path=selected_path,
            uncertainty_score=u_ctx,
            remaining_latency_budget_ms=remaining_budget,
            routing_latency_ms=routing_time_ms,
            rationale=reason
        )
