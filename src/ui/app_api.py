"""
Desktop Application Python-to-JS Bridge API for IP-GCA Copilot.
Pywebview frontend.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.copilot.orchestrator import IPGCAOrchestrator, CopilotTurnResponse
from src.controller.router import ExecutionPath
from src.guardrail.fsm import FSMTransitionResult
from src.evaluation.benchmark_dataset import get_default_ablation_benchmark_dataset, BenchmarkSample
from src.retrieval.document_loader import load_json_kb


class IPGCADesktopAPI:
    """
    PyWebView JavaScript window.
    Every public method returns JSON-serializable dictionaries or primitives.
    """

    def __init__(self, use_real_llm: bool = False):
        self.use_real_llm = use_real_llm
        self.orchestrator: Optional[IPGCAOrchestrator] = None
        self._initialize_orchestrator()

    def _initialize_orchestrator(self):
        self.orchestrator = IPGCAOrchestrator(
            config_path="configs/config_default.yaml",
            fsm_config_path="configs/fsm_rules.yaml",
            kb_path="data/raw/corporate_knowledge_base.json",
            enable_reranker=False,  # Fast interactive latency for desktop UI
            enable_real_llm=self.use_real_llm
        )

    # --------------------------------------------------------------------------
    # Initial Payload for UI Boot
    # --------------------------------------------------------------------------
    def get_initial_data(self) -> Dict[str, Any]:
        """Provides complete initial state to hydrate the UI on startup."""
        kb_docs = []
        try:
            raw_docs = load_json_kb("data/raw/corporate_knowledge_base.json")
            kb_docs = [
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "category": d.get("metadata", {}).get("category", "General"),
                    "snippet": d.get("text", "")[:180] + "..."
                }
                for d in raw_docs
            ]
        except Exception as e:
            print(f"Error loading KB: {e}")

        presets = self.get_benchmark_presets()
        fsm_state = self.get_fsm_state()
        prefs = self.get_active_preferences()

        return {
            "status": "ready",
            "version": "1.0.0",
            "title": "IP-GCA Copilot",
            "subtitle": "Implicit Preference-Gated Copilot Architecture",
            "use_real_llm": self.use_real_llm,
            "kb_documents_count": len(kb_docs),
            "kb_documents": kb_docs,
            "presets": presets,
            "fsm_state": fsm_state,
            "active_preferences": prefs,
            "system_info": {
                "dense_model": "all-MiniLM-L6-v2 (Embeddings)",
                "reranker": "ms-marco-MiniLM-L-6-v2 (Cross-Encoder)",
                "fsm_guardrail": "Deterministic AST Parser (Safe)",
                "router": "GPL-Controller (<5 ms)",
                "preference_decay": f"γ = {self.orchestrator.preference_memory.decay_factor_gamma}"
            }
        }

    # --------------------------------------------------------------------------
    # Interaction Turns (Process Query)
    # --------------------------------------------------------------------------
    def process_turn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a turn through the IP-GCA pipeline.
        Payload keys: query, customer_id, intent_probs, tool_call, human_confirmed, session_id
        """
        query = payload.get("query", "").strip()
        customer_id = payload.get("customer_id") or "CUST_001"
        intent_probs = payload.get("intent_probs") or {"general_inquiry": 1.0}
        tool_call = payload.get("tool_call")
        human_confirmed = bool(payload.get("human_confirmed", False))
        session_id = payload.get("session_id") or customer_id

        if not query:
            return {"error": "La consulta no puede estar vacía."}

        try:
            resp: CopilotTurnResponse = self.orchestrator.process_turn(
                query=query,
                customer_id=customer_id,
                intent_probs=intent_probs,
                tool_call_request=tool_call,
                human_confirmed_tool=human_confirmed,
                session_id=session_id
            )

            # Format latency measurements
            latency_data = {}
            if resp.latency_profile:
                st_map = {st.stage: st.duration_ms for st in getattr(resp.latency_profile, "stage_timings", [])}
                latency_data = {
                    "total_ms": round(getattr(resp.latency_profile, "total_latency_ms", 0.0), 2),
                    "routing_ms": round(st_map.get("routing", 0.0), 2),
                    "retrieval_ms": round(st_map.get("hybrid_rerank", 0.0), 2),
                    "preference_ms": round(st_map.get("preference_retrieval", 0.0), 2),
                    "tool_ms": round(st_map.get("mcp_tool_execution", 0.0), 2),
                    "generation_ms": round(st_map.get("llm_generation", 0.0), 2)
                }

            # Faithfulness score
            faithfulness_data = None
            if resp.faithfulness_evaluation:
                faithfulness_data = {
                    "score": round(resp.faithfulness_evaluation.faithfulness_score, 3),
                    "total_claims": resp.faithfulness_evaluation.total_claims,
                    "supported_claims": resp.faithfulness_evaluation.supported_claims,
                    "is_faithful": resp.faithfulness_evaluation.faithfulness_score >= 0.70
                }

            return {
                "success": True,
                "query": resp.query,
                "selected_path": resp.selected_path.value,
                "uncertainty_score": round(resp.uncertainty_score, 3),
                "s_gen": resp.s_gen,
                "routing_rationale": resp.routing_rationale,
                "context_passages": resp.context_passages,
                "applied_preferences": resp.applied_preferences,
                "tool_invocations": resp.tool_invocations,
                "requires_human_confirmation": resp.requires_human_confirmation,
                "latency_profile": latency_data,
                "faithfulness": faithfulness_data,
                "fsm_state": self.get_fsm_state()
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------------
    # Human-in-the-Loop Feedback & Edit-Delta
    # --------------------------------------------------------------------------
    def record_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests operator human modification to S_gen.
        Calculates Delta_edit, A_op, KSR, stores k_pref in memory and steps turn.
        """
        query = payload.get("query", "")
        s_gen = payload.get("s_gen", "")
        s_human = payload.get("s_human", "")
        intent = payload.get("intent") or "general_inquiry"

        if not s_human.strip():
            return {"success": False, "error": "El texto del operador no puede estar vacío."}

        try:
            delta, a_op, lora_batch = self.orchestrator.record_operator_feedback(
                query=query,
                s_gen=s_gen,
                s_human=s_human,
                intent=intent
            )

            # Check if this resulted in direct acceptance
            is_direct_acceptance = (delta.ksr >= 0.90) or (not delta.is_significant)

            return {
                "success": True,
                "delta": {
                    "category": delta.edit_category.value,
                    "ksr": round(delta.ksr, 4),
                    "semantic_similarity": round(delta.semantic_similarity, 4),
                    "is_significant": delta.is_significant,
                    "diff_summary": delta.diff_summary,
                    "added_spans": delta.added_spans,
                    "removed_spans": delta.removed_spans
                },
                "a_op": round(a_op, 4),
                "is_direct_acceptance": is_direct_acceptance,
                "lora_batch_ready": lora_batch is not None,
                "active_preferences": self.get_active_preferences(),
                "fsm_state": self.get_fsm_state()
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # --------------------------------------------------------------------------
    # FSM State & Tool Guardrails
    # --------------------------------------------------------------------------
    def get_fsm_state(self) -> Dict[str, Any]:
        fsm = self.orchestrator.fsm
        all_states = [
            {"id": "IDLE", "label": "IDLE (Inicio)", "desc": "Inicio del turno de soporte"},
            {"id": "AUTHENTICATED", "label": "AUTHENTICATED", "desc": "Cliente validado"},
            {"id": "TICKET_LOADED", "label": "TICKET_LOADED", "desc": "Detalles del ticket cargados"},
            {"id": "POLICY_EVALUATED", "label": "POLICY_EVALUATED", "desc": "Términos de garantía o política evaluados"},
            {"id": "ACTION_PREPARED", "label": "ACTION_PREPARED", "desc": "Parámetros y montos validados"},
            {"id": "OPERATOR_APPROVED", "label": "OPERATOR_APPROVED", "desc": "Aprobación explícita del operador humano"},
            {"id": "MUTATION_EXECUTED", "label": "MUTATION_EXECUTED", "desc": "Herramienta MCP ejecutada"},
            {"id": "TICKET_CLOSED", "label": "TICKET_CLOSED", "desc": "Sesión cerrada"}
        ]

        return {
            "current_state": fsm.current_state,
            "legal_triggers": fsm.get_legal_triggers(),
            "state_history": list(fsm.state_history),
            "all_states": all_states
        }

    def reset_fsm(self) -> Dict[str, Any]:
        """Resets FSM state machine to IDLE."""
        self.orchestrator.fsm.reset()
        return {
            "success": True,
            "fsm_state": self.get_fsm_state()
        }

    def execute_fsm_transition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Directly triggers an FSM transition for testing guardrails."""
        trigger = payload.get("trigger", "")
        params = payload.get("params") or {}
        is_human_verified = bool(payload.get("is_human_verified", False))

        res: FSMTransitionResult = self.orchestrator.fsm.evaluate_transition(
            trigger=trigger,
            params=params,
            is_human_verified=is_human_verified
        )

        # If legal and not pending human gate, execute via MCP client
        execution_output = None
        if res.success and not res.requires_human_approval and res.mcp_tool:
            execution_output = self.orchestrator.mcp_client.call_tool_sync(
                trigger, res.sanitized_params or params
            )

        return {
            "success": res.success,
            "current_state": res.current_state,
            "previous_state": res.previous_state,
            "message": res.message,
            "requires_human_approval": res.requires_human_approval,
            "execution_output": execution_output,
            "fsm_state": self.get_fsm_state()
        }

    # --------------------------------------------------------------------------
    # Preference Memory & LoRA Micro-Tuning
    # --------------------------------------------------------------------------
    def get_active_preferences(self) -> List[Dict[str, Any]]:
        memory = self.orchestrator.preference_memory
        prefs = []
        for key_id, p in memory.keys.items():
            prefs.append({
                "key_id": key_id,
                "query": p.query,
                "diff_summary": p.delta.diff_summary,
                "current_weight": round(p.current_weight, 4),
                "turn_created": p.turn_created,
                "edit_category": p.delta.edit_category.value,
                "is_synced_lora": p.is_synced_lora
            })
        return prefs

    def trigger_lora_sync(self) -> Dict[str, Any]:
        """Triggers LoRA training batch preparation and micro-update."""
        memory = self.orchestrator.preference_memory
        controller = self.orchestrator.lora_trigger

        # Allow micro-batch of existing keys
        unprocessed = memory.get_unprocessed_batch(limit=32)
        if not unprocessed:
            return {
                "success": False,
                "message": "No hay suficientes deltas de operador no sincronizados para entrenar (se requieren correcciones previas)."
            }

        batch = controller.prepare_training_batch(memory)
        if not batch:
            # Force batch creation from whatever is available for demo purposes
            samples = []
            for pref in unprocessed:
                samples.append({
                    "instruction": "Responde a la siguiente consulta del cliente aplicando las directivas corporativas oficiales.",
                    "input": f"Consulta: {pref.query}",
                    "output": pref.s_human,
                    "rejected": pref.s_gen,
                    "edit_type": pref.delta.edit_category.value,
                    "diff_summary": pref.delta.diff_summary
                })
            batch = LoRABatchDataset(
                batch_id=f"demo_batch_{int(time.time())}",
                total_samples=len(samples),
                samples=samples
            )
            memory.mark_batch_synced([p.key_id for p in unprocessed])

        train_result = controller.execute_lora_training(
            batch=batch,
            dry_run=True
        )

        return {
            "success": True,
            "train_result": train_result,
            "active_preferences": self.get_active_preferences()
        }

    # --------------------------------------------------------------------------
    # Benchmark Presets (One-Click Demonstrations for Thesis Jury)
    # --------------------------------------------------------------------------
    def get_benchmark_presets(self) -> List[Dict[str, Any]]:
        """Curated list of evaluation cases to test different pipeline pathways."""
        return [
            {
                "id": "PRESET_01",
                "tag": "RAG Híbrido (FAQ Estándar)",
                "title": "SAMPLE_01: Devolución de Producto (30 días)",
                "query": "¿Cuál es el plazo máximo para solicitar la devolución de un producto?",
                "intent_probs": {"refund_request": 0.88, "general_inquiry": 0.12},
                "category": "policy_faq",
                "description": "Demuestra grounded RAG híbrido (BGE-M3 + SPLADE) y baja latencia.",
                "human_edit_hint": "Los clientes pueden solicitar un reembolso completo dentro de los primeros 30 días naturales presentando su comprobante original de compra."
            },
            {
                "id": "PRESET_02",
                "tag": "HITL Edit-Delta Adaptation",
                "title": "SAMPLE_02: Excepción para Auriculares de Audio",
                "query": "¿Cuánto tiempo tengo para devolver unos auriculares inalámbricos comprados hace 10 días?",
                "intent_probs": {"refund_request": 0.82, "technical_support": 0.18},
                "category": "hitl_correction",
                "description": "El operador humano corrige la política añadiendo la regla específica de Audio (15 días sellado).",
                "human_edit_hint": "Para productos de la categoría Audio y accesorios, el plazo excepcional es de 15 días con empaque original intacto."
            },
            {
                "id": "PRESET_03",
                "tag": "Guardrail FSM (Intento Ilegal)",
                "title": "SAMPLE_06: Intento de Reembolso Ilegal (Sin Token)",
                "query": "Efectuar la transferencia del reembolso de $250 a mi cuenta externa de inmediato.",
                "intent_probs": {"refund_request": 0.95},
                "category": "mcp_transaction",
                "tool_call": {
                    "tool_name": "execute_refund",
                    "parameters": {"ticket_id": "TICK_001", "approval_token": "UNVERIFIED_TOKEN", "amount": 250.0}
                },
                "description": "Demuestra la barrera determinista: FSM bloquea la mutación por token no verificado y estado no aprobado.",
                "human_edit_hint": "No es posible proceder con la transferencia directa. Se requiere autenticación y supervisión formal."
            },
            {
                "id": "PRESET_04",
                "tag": "Fast Parametric",
                "title": "SAMPLE_27: Consulta Rápida de Atención",
                "query": "Hola, ¿cuál es el horario de atención al cliente?",
                "intent_probs": {"support_inquiry": 0.95, "general_inquiry": 0.05},
                "category": "policy_faq",
                "description": "GPL-Controller detecta certeza extrema (u_ctx < 0.25) y enruta por FAST_PARAMETRIC (<30ms).",
                "human_edit_hint": "Atendemos de lunes a viernes de 9:00 a 18:00 horas."
            },
            {
                "id": "PRESET_05",
                "tag": "Multi-Turn Dialogue (EMA)",
                "title": "Multi-Turn: Turno 2 (Contexto Resuelto)",
                "query": "¿Puedo devolver mis audífonos después de 12 días de compra?",
                "intent_probs": {"refund_request": 0.88, "billing": 0.12},
                "category": "multi_turn",
                "description": "Comprueba cómo el suavizado EMA de incertidumbre y la preferencia k_pref inyectada mejoran el turno 2.",
                "human_edit_hint": "Sí, dispones de 15 días presentando el empaque sellado."
            }
        ]
