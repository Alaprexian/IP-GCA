"""
IP-GCA End-to-End Copilot Orchestrator.
Unifies:
1. Dynamic routing under joint uncertainty & latency constraints (GPL-Controller).
2. Two-stage hybrid retrieval (BGE-M3 + SPLADE + Cross-Encoder).
3. Online edit-delta preference adaptation (Delta_edit -> k_pref).
4. Deterministic FSM guardrail isolation for MCP tool execution.
5. Non-circular evaluation & latency profiling (Le2e, F_faith, A_op, A_tool).
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import time

from src.controller.router import GPLController, ExecutionPath, RoutingDecision
from src.retrieval.engine import HybridRetrievalEngine
from src.retrieval.document_loader import build_index_from_file
from src.guardrail.fsm import TransactionalGuardrailFSM, FSMTransitionResult
from src.guardrail.mcp_client import MCPClient
from src.adaptation.edit_extractor import EditDeltaExtractor, SemanticDelta
from src.adaptation.preference_memory import PreferenceMemory, PreferenceKey
from src.adaptation.conditioning_bridge import EditDeltaConditioningBridge
from src.adaptation.lora_trigger import LoRATriggerController, LoRABatchDataset
from src.evaluation.latency_tracker import LatencyTracker, RequestProfile
from src.evaluation.alignment import calculate_operator_alignment
from src.evaluation.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.copilot.generator import LLMGenerator



class CopilotTurnResponse(BaseModel):
    query: str
    selected_path: ExecutionPath
    s_gen: str
    uncertainty_score: float
    context_passages: List[str]
    applied_preferences: List[str]
    tool_invocations: List[Dict[str, Any]] = Field(default_factory=list)
    requires_human_confirmation: bool = False
    routing_rationale: str
    latency_profile: Optional[RequestProfile] = None
    faithfulness_evaluation: Optional[FaithfulnessResult] = None


class IPGCAOrchestrator:
    """
    Unified IP-GCA Copilot orchestrating multi-loop customer support interactions.
    """

    def __init__(
        self,
        config_path: str = "configs/config_default.yaml",
        fsm_config_path: str = "configs/fsm_rules.yaml",
        kb_path: Optional[str] = "data/raw/corporate_knowledge_base.json",
        enable_reranker: bool = True,
        enable_real_llm: bool = False,
        llm_model_path: Optional[str] = None,
        dense_model_override: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_override: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.config_path = config_path
        self.controller = GPLController(config_path)
        self.fsm = TransactionalGuardrailFSM(fsm_config_path)
        self.mcp_client = MCPClient()

        self.edit_extractor = EditDeltaExtractor()
        self.preference_memory = PreferenceMemory(config_path)
        self.conditioning_bridge = EditDeltaConditioningBridge()
        self.lora_trigger = LoRATriggerController(config_path)

        self.latency_tracker = LatencyTracker()
        self.faithfulness_evaluator = FaithfulnessEvaluator(load_model=False)

        self.enable_real_llm = enable_real_llm
        self.generator: Optional[LLMGenerator] = None
        if enable_real_llm:
            self.generator = LLMGenerator(
                model_name_or_path=llm_model_path or "models/Qwen2.5-1.5B-Instruct",
                load_in_4bit=True
            )

        self.engine = HybridRetrievalEngine(
            enable_reranker=enable_reranker,
            dense_model_override=dense_model_override,
            reranker_model_override=reranker_model_override
        )

        if kb_path:
            try:
                build_index_from_file(kb_path, self.engine)
            except Exception:
                pass

    def process_turn(
        self,
        query: str,
        customer_id: Optional[str] = None,
        intent_probs: Optional[Dict[str, float]] = None,
        tool_call_request: Optional[Dict[str, Any]] = None,
        human_confirmed_tool: bool = False,
        session_id: Optional[str] = None
    ) -> CopilotTurnResponse:
        """
        Executes a complete customer support interaction turn through the IP-GCA pipeline.
        """
        # Reset FSM to IDLE at the start of each new interaction turn
        # to prevent state leakage across consecutive customer turns.
        self.fsm.reset()
        self.latency_tracker.start_request(query)
        intents = intent_probs or {"general_inquiry": 1.0}
        active_session = session_id or customer_id

        with self.latency_tracker.measure("routing"):
            pre_results = self.engine.retrieve(query, top_k=5, use_reranker=False)
            retrieval_scores = self.engine.extract_scores(pre_results)
            decision = self.controller.route_query(
                query,
                retrieval_scores,
                intents,
                session_id=active_session
            )

        context_passages: List[str] = []
        tool_results: List[Dict[str, Any]] = []
        requires_human_approval = False

        with self.latency_tracker.measure("preference_retrieval"):
            relevant_prefs = self.preference_memory.retrieve_relevant(query, top_k=3)
            applied_pref_labels = [
                f"{p[0].delta.diff_summary} (w={p[1]:.2f})"
                for p in relevant_prefs
            ]

        if tool_call_request:
            with self.latency_tracker.measure("mcp_tool_execution"):
                tool_name = tool_call_request.get("tool_name", "")
                params = tool_call_request.get("parameters", {})

                fsm_res: FSMTransitionResult = self.fsm.evaluate_transition(
                    trigger=tool_name,
                    params=params,
                    is_human_verified=human_confirmed_tool
                )

                if fsm_res.success:
                    if fsm_res.requires_human_approval and not human_confirmed_tool:
                        requires_human_approval = True
                        tool_results.append({
                            "status": "PENDING_HUMAN_APPROVAL",
                            "tool": tool_name,
                            "message": fsm_res.message,
                            "params": params
                        })
                    else:
                        res = self.mcp_client.call_tool_sync(tool_name, fsm_res.sanitized_params or params)
                        tool_results.append({
                            "status": "EXECUTED",
                            "tool": tool_name,
                            "result": res
                        })
                else:
                    tool_results.append({
                        "status": "GUARDRAIL_BLOCKED",
                        "tool": tool_name,
                        "error": fsm_res.message
                    })

        if (decision.selected_path == ExecutionPath.FAST_PARAMETRIC) and not tool_call_request:
            if pre_results:
                context_passages = [pre_results[0].chunk.content]
            s_gen = self._generate_parametric(query, context_passages, relevant_prefs)

        elif (decision.selected_path == ExecutionPath.HYBRID_RAG) and not tool_call_request:
            with self.latency_tracker.measure("hybrid_rerank"):
                final_results = self.engine.retrieve(query, top_k=3, use_reranker=True)
                context_passages = [r.chunk.content for r in final_results]

            s_gen = self._generate_grounded(query, context_passages, relevant_prefs)

        elif decision.selected_path == ExecutionPath.AGENTIC_REFLECTION or tool_call_request:
            with self.latency_tracker.measure("hybrid_rerank"):
                final_results = self.engine.retrieve(query, top_k=3, use_reranker=True)
                context_passages = [r.chunk.content for r in final_results]

            s_gen = self._generate_agentic(query, context_passages, tool_results, relevant_prefs)

        else: # ExecutionPath.ASSISTED_ABSTENTION
            s_gen = self._generate_abstention_package(query, decision.uncertainty_score, pre_results)


        profile = self.latency_tracker.finish_request(result_count=len(context_passages))

        faith_res = None
        if context_passages and decision.selected_path in [ExecutionPath.FAST_PARAMETRIC, ExecutionPath.HYBRID_RAG, ExecutionPath.AGENTIC_REFLECTION]:
            faith_res = self.faithfulness_evaluator.evaluate(s_gen, context_passages)

        return CopilotTurnResponse(
            query=query,
            selected_path=decision.selected_path,
            s_gen=s_gen,
            uncertainty_score=decision.uncertainty_score,
            context_passages=context_passages,
            applied_preferences=applied_pref_labels,
            tool_invocations=tool_results,
            requires_human_confirmation=requires_human_approval,
            routing_rationale=decision.rationale,
            latency_profile=profile,
            faithfulness_evaluation=faith_res
        )

    def record_operator_feedback(
        self,
        query: str,
        s_gen: str,
        s_human: str,
        intent: Optional[str] = None
    ) -> Tuple[SemanticDelta, float, Optional[LoRABatchDataset]]:
        """
        Feedback ingestion loop:
        1. Extracts delta Delta_edit between suggestion and human correction.
        2. Computes operator alignment A_op.
        3. Updates preference memory with recency decay gamma.
        4. Triggers LoRA dataset preparation if batch threshold (32 deltas) is reached.
        """
        delta = self.edit_extractor.extract(s_gen, s_human, query=query)
        a_op = calculate_operator_alignment(s_gen, s_human)

        if delta.is_significant:
            self.preference_memory.add_preference(
                query=query,
                s_gen=s_gen,
                s_human=s_human,
                delta=delta,
                intent=intent
            )

        self.preference_memory.step_turn()

        lora_batch = None
        if self.lora_trigger.is_threshold_reached(self.preference_memory):
            lora_batch = self.lora_trigger.prepare_training_batch(self.preference_memory)

        return delta, a_op, lora_batch

    def _generate_parametric(
        self,
        query: str,
        passages: List[str],
        preferences: List[Tuple[PreferenceKey, float]]
    ) -> str:
        if self.generator and self.generator.is_loaded:
            with self.latency_tracker.measure("llm_generation"):
                sys_prompt = (
                    "Eres el asistente oficial de soporte de la empresa (IP-GCA Copilot). "
                    "Responde a la consulta de forma breve, concisa y cordial en una o dos oraciones."
                )
                return self.generator.generate(
                    prompt=query,
                    system_prompt=sys_prompt,
                    max_new_tokens=80,
                    temperature=0.1
                )

        if passages:
            first_line = passages[0].split("\n")[0]
            return f"Según nuestras políticas: {first_line}. Si necesitas asistencia adicional con gusto te ayudamos."
        return f"Hemos recibido tu consulta sobre '{query}'. Nuestro equipo procesará tu solicitud de inmediato."

    def _generate_grounded(
        self,
        query: str,
        passages: List[str],
        preferences: List[Tuple[PreferenceKey, float]]
    ) -> str:
        if self.generator and self.generator.is_loaded:
            with self.latency_tracker.measure("llm_generation"):
                messages = self.conditioning_bridge.assemble_chat_messages(
                    user_query=query,
                    retrieved_passages=passages,
                    ranked_preferences=preferences
                )
                return self.generator.generate(
                    messages=messages,
                    max_new_tokens=180,
                    temperature=0.1
                )

        # Deterministic grounded synthesis fallback
        lead_passage = passages[0] if passages else "Políticas oficiales de la empresa."
        pref_addon = ""
        if preferences:
            top_pref = preferences[0][0]
            if top_pref.delta.added_spans:
                pref_addon = f" Nota importante: {', '.join(top_pref.delta.added_spans)}."

        return f"De acuerdo a nuestra política oficial: {lead_passage[:180].strip()}...{pref_addon}"


    def _generate_agentic(
        self,
        query: str,
        passages: List[str],
        tools: List[Dict[str, Any]],
        preferences: List[Tuple[PreferenceKey, float]]
    ) -> str:
        tool_summary = ""
        for t in tools:
            status = t.get("status")
            tool_name = t.get("tool")
            if status == "EXECUTED":
                tool_summary += f" [Operación '{tool_name}' ejecutada exitosamente]"
            elif status == "PENDING_HUMAN_APPROVAL":
                tool_summary += f" [Operación crítica '{tool_name}' requiere confirmación humana]"
            elif status == "GUARDRAIL_BLOCKED":
                tool_summary += f" [Operación '{tool_name}' bloqueada por guardrail: {t.get('error')}]"

        base = self._generate_grounded(query, passages, preferences)
        return f"{base}{tool_summary}"

    def _generate_abstention_package(
        self,
        query: str,
        uncertainty: float,
        pre_results: List[Any]
    ) -> str:
        return (
            f"[DERIVACIÓN ASISTIDA A OPERADOR HUMANO]\n"
            f"- Motivo: Incertidumbre contextual elevada (u_ctx = {uncertainty:.3f} >= 0.85).\n"
            f"- Consulta del cliente: \"{query}\"\n"
            f"- Pasajes ambiguos encontrados: {len(pre_results)}\n"
            f"- Acción recomendada: Un agente humano debe evaluar la situación y responder manualmente."
        )
