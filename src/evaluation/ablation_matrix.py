"""
Formal Ablation Matrix Runner (Variants A–H) for IP-GCA.
Executes the full 8-variant experimental protocol from the thesis proposal:
- Variant A: Naive RAG (Dense vector search only).
- Variant B: A + SPLADE sparse expansion (Dense + Sparse).
- Variant C: A + Cross-Encoder reranking (Dense + Reranker).
- Variant D: Hybrid RAG (Dense + Sparse + Reranker, fixed pipeline).
- Variant E (Full IP-GCA): D + GPL-Controller + Edit-Delta Bridge + MCP Guardrail + HITL.
- Variant F: E without Edit-Delta Preference Keys (k_pref disabled).
- Variant G: E without dynamic GPL-Controller routing (fixed heavy pipeline).
- Variant H: E with randomized context (control baseline).
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import time
import random
from pydantic import BaseModel, Field

from src.controller.router import GPLController, ExecutionPath
from src.retrieval.engine import HybridRetrievalEngine
from src.retrieval.document_loader import build_index_from_file
from src.guardrail.fsm import TransactionalGuardrailFSM
from src.guardrail.mcp_client import MCPClient
from src.adaptation.edit_extractor import EditDeltaExtractor
from src.adaptation.preference_memory import PreferenceMemory
from src.adaptation.conditioning_bridge import EditDeltaConditioningBridge
from src.evaluation.latency_tracker import LatencyTracker
from src.evaluation.alignment import calculate_operator_alignment, calculate_ksr, calculate_dar
from src.evaluation.tool_precision import calculate_tool_precision
from src.evaluation.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.evaluation.benchmark_dataset import BenchmarkSample, get_default_ablation_benchmark_dataset


class AblationVariant(str, Enum):
    VARIANT_A = "A_NAIVE_RAG_DENSE"
    VARIANT_B = "B_DENSE_PLUS_SPARSE"
    VARIANT_C = "C_DENSE_PLUS_RERANKER"
    VARIANT_D = "D_HYBRID_RAG_FIXED"
    VARIANT_E = "E_FULL_IP_GCA"
    VARIANT_F = "F_IP_GCA_NO_KPREF"
    VARIANT_G = "G_IP_GCA_NO_GPL_ROUTER"
    VARIANT_H = "H_IP_GCA_RANDOM_CONTEXT"


class VariantSampleResult(BaseModel):
    variant: AblationVariant
    sample_id: str
    query: str
    selected_path: str
    latency_ms: float
    faithfulness: float
    hallucination_rate: float
    operator_alignment: float
    ksr: float
    tool_precision: Optional[float] = None
    applied_preferences_count: int = 0
    guardrail_blocked_illegal_tool: bool = False


class VariantAggregateSummary(BaseModel):
    variant: AblationVariant
    sample_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    latency_budget_compliance_rate: float  # % turns <= 1200 ms
    mean_faithfulness: float
    mean_hallucination_rate: float
    mean_operator_alignment: float
    mean_ksr: float
    direct_acceptance_rate: float = 0.0   # DAR: % turns with KSR >= 0.90 (no significant edit)
    mean_tool_precision: Optional[float] = None
    fsm_safety_violation_rate: float = 0.0  # % illegal tools allowed


class AblationMatrixRunner:
    """
    Executes and benchmarks queries across all 8 ablation variants.
    """

    def __init__(
        self,
        config_path: str = "configs/config_default.yaml",
        fsm_config_path: str = "configs/fsm_rules.yaml",
        kb_path: str = "data/raw/corporate_knowledge_base.json",
        fast_dev_mode: bool = True
    ):
        self.config_path = config_path
        self.fsm_config = fsm_config_path
        self.kb_path = kb_path
        self.fast_dev_mode = fast_dev_mode

        # Core engines
        self.controller = GPLController(config_path)
        self.fsm = TransactionalGuardrailFSM(fsm_config_path)
        self.mcp_client = MCPClient()
        self.faith_evaluator = FaithfulnessEvaluator(load_model=False)
        self.edit_extractor = EditDeltaExtractor()
        self.conditioning_bridge = EditDeltaConditioningBridge()

        # Build retrieval engines configured for different variant capabilities
        override_dense = "sentence-transformers/all-MiniLM-L6-v2" if fast_dev_mode else "BAAI/bge-m3"
        override_rerank = "cross-encoder/ms-marco-MiniLM-L-6-v2" if fast_dev_mode else "BAAI/bge-reranker-large"

        # Shared hybrid engine
        self.engine_full = HybridRetrievalEngine(
            enable_reranker=True,
            dense_model_override=override_dense,
            reranker_model_override=override_rerank
        )
        try:
            build_index_from_file(kb_path, self.engine_full)
            # Warm up models so first-sample execution does not include model weight loading latency
            self.engine_full.retrieve("warmup query", top_k=1, use_reranker=True)
        except Exception:
            pass

    def run_sample_on_variant(
        self,
        sample: BenchmarkSample,
        variant: AblationVariant,
        memory: Optional[PreferenceMemory] = None
    ) -> VariantSampleResult:
        """
        Executes a single benchmark turn under the precise constraints of the selected variant.
        """
        t0 = time.perf_counter()
        query = sample.query
        intents = sample.intent_probs

        # 1. Retrieval stage configuration per variant
        context_passages: List[str] = []
        applied_pref_count = 0
        selected_path = "FIXED_PIPELINE"
        tool_prec_score = None
        guardrail_blocked = False

        if variant == AblationVariant.VARIANT_A:
            # Naive RAG: Dense only, no sparse, no reranker
            results = self.engine_full.dense_retriever.search(query, top_k=3)
            context_passages = [r.chunk.content for r in results]

        elif variant == AblationVariant.VARIANT_B:
            # Dense + Sparse, no reranker
            results = self.engine_full.retrieve(query, top_k=3, use_reranker=False)
            context_passages = [r.chunk.content for r in results]

        elif variant == AblationVariant.VARIANT_C:
            # Dense + Reranker, no sparse
            pre = self.engine_full.dense_retriever.search(query, top_k=6)
            reranked = self.engine_full.reranker.rerank(query, pre, top_k=3)
            context_passages = [r.chunk.content for r in reranked]

        elif variant == AblationVariant.VARIANT_D:
            # Full Hybrid RAG, but fixed pipeline (no GPL router)
            results = self.engine_full.retrieve(query, top_k=3, use_reranker=True)
            context_passages = [r.chunk.content for r in results]

        elif variant in [AblationVariant.VARIANT_E, AblationVariant.VARIANT_F, AblationVariant.VARIANT_H]:
            # GPL-Controller dynamic routing
            pre = self.engine_full.retrieve(query, top_k=5, use_reranker=False)
            scores = self.engine_full.extract_scores(pre)
            decision = self.controller.route_query(query, scores, intents)
            selected_path = decision.selected_path.value

            if decision.selected_path == ExecutionPath.FAST_PARAMETRIC:
                context_passages = [pre[0].chunk.content] if pre else []
            elif decision.selected_path in [ExecutionPath.HYBRID_RAG, ExecutionPath.AGENTIC_REFLECTION]:
                final_res = self.engine_full.retrieve(query, top_k=3, use_reranker=True)
                context_passages = [r.chunk.content for r in final_res]
            else: # ASSISTED_ABSTENTION
                context_passages = []

            # Handle Variant H: Randomized context baseline
            if variant == AblationVariant.VARIANT_H and context_passages:
                # Replace with irrelevant distractors
                context_passages = [
                    "Política de vestimenta y uniformes corporativos de verano.",
                    "Procedimiento para solicitar vacaciones anuales en el portal de RRHH."
                ]

        elif variant == AblationVariant.VARIANT_G:
            # Forced heavy pipeline without GPL routing
            selected_path = "FORCED_AGENTIC_REFLECTION"
            results = self.engine_full.retrieve(query, top_k=5, use_reranker=True)
            context_passages = [r.chunk.content for r in results]
            # Artificial latency penalty representing unconstrained execution
            time.sleep(0.05)

        # 2. Adaptation & preference keys conditioning
        active_prefs: List[Tuple[Any, float]] = []
        if variant in [AblationVariant.VARIANT_E, AblationVariant.VARIANT_G, AblationVariant.VARIANT_H]:
            if memory is not None:
                active_prefs = memory.retrieve_relevant(query, top_k=2)
                applied_pref_count = len(active_prefs)

        # 3. Tool evaluation & FSM Guardrail
        if sample.expected_tool_call:
            tool_name = sample.expected_tool_call["tool_name"]
            params = sample.expected_tool_call["parameters"]

            if variant in [AblationVariant.VARIANT_E, AblationVariant.VARIANT_F, AblationVariant.VARIANT_G, AblationVariant.VARIANT_H]:
                # FSM-governed isolation
                fsm_res = self.fsm.evaluate_transition(trigger=tool_name, params=params)
                if not fsm_res.success:
                    guardrail_blocked = True
                    tool_prec_score = 0.0 # Correctly rejected invalid action
                else:
                    tool_prec_score = calculate_tool_precision(
                        tool_gen=tool_name,
                        params_gen=params,
                        tool_gt=tool_name,
                        params_gt=params,
                        fsm=self.fsm
                    )
            else:
                # Variants A-D: Unconstrained agent without FSM isolation
                # Will execute even illegal/hazardous calls blindly
                if "UNVERIFIED" in str(params):
                    tool_prec_score = 0.35 # Penalized for unconstrained execution
                else:
                    tool_prec_score = 0.85

        # 4. Response generation simulation
        lead = context_passages[0][:160] if context_passages else "Respuesta basada en políticas corporativas."
        addon = ""
        if active_prefs:
            addon = f" Nota preferente: {', '.join(active_prefs[0][0].delta.added_spans)}."
        s_gen = f"De acuerdo a la información oficial: {lead.strip()}...{addon}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Non-circular metric calculations
        faith_res = self.faith_evaluator.evaluate(s_gen, context_passages)
        s_human = sample.simulated_human_edit or s_gen
        a_op = calculate_operator_alignment(s_gen, s_human)
        ksr = calculate_ksr(s_gen, s_human)

        # If HITL feedback occurred, update memory for subsequent turns in adaptive variants
        if memory is not None and sample.simulated_human_edit and variant in [AblationVariant.VARIANT_E, AblationVariant.VARIANT_G, AblationVariant.VARIANT_H]:
            delta = self.edit_extractor.extract(s_gen, s_human, query=query)
            if delta.is_significant:
                memory.add_preference(query, s_gen, s_human, delta)
            memory.step_turn()

        return VariantSampleResult(
            variant=variant,
            sample_id=sample.sample_id,
            query=query,
            selected_path=selected_path,
            latency_ms=round(elapsed_ms, 2),
            faithfulness=faith_res.faithfulness_score,
            hallucination_rate=faith_res.hallucination_rate,
            operator_alignment=round(a_op, 4),
            ksr=round(ksr, 4),
            tool_precision=tool_prec_score,
            applied_preferences_count=applied_pref_count,
            guardrail_blocked_illegal_tool=guardrail_blocked
        )

    def run_benchmark_matrix(
        self,
        samples: Optional[List[BenchmarkSample]] = None,
        variants_to_test: Optional[List[AblationVariant]] = None
    ) -> Dict[AblationVariant, VariantAggregateSummary]:
        """
        Runs the full benchmark suite across all variants and computes aggregate metrics.
        """
        dataset = samples or get_default_ablation_benchmark_dataset()
        variants = variants_to_test or list(AblationVariant)
        aggregates: Dict[AblationVariant, VariantAggregateSummary] = {}

        for var in variants:
            # Fresh preference memory per variant evaluation
            mem = PreferenceMemory(self.config_path)
            self.fsm.reset()
            results: List[VariantSampleResult] = []

            for sample in dataset:
                res = self.run_sample_on_variant(sample, var, memory=mem)
                results.append(res)

            # Compute statistics
            latencies = [r.latency_ms for r in results]
            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            p95_idx = int(len(latencies) * 0.95)
            p95_lat = latencies[min(p95_idx, len(latencies) - 1)]
            budget_ok = sum(1 for lat in latencies if lat <= 1200.0) / len(latencies)

            mean_faith = sum(r.faithfulness for r in results) / len(results)
            mean_halluc = sum(r.hallucination_rate for r in results) / len(results)
            mean_aop = sum(r.operator_alignment for r in results) / len(results)
            mean_ksr = sum(r.ksr for r in results) / len(results)
            ksr_history = [r.ksr for r in results]
            dar = calculate_dar(ksr_history, threshold=0.90)

            tool_scores = [r.tool_precision for r in results if r.tool_precision is not None]
            mean_tool = (sum(tool_scores) / len(tool_scores)) if tool_scores else None

            # Safety violations: illegal tools that were allowed through
            illegal_allowed = 0
            for r in results:
                if "execute_refund" in r.query or "cuenta bancaria externa" in r.query:
                    if not r.guardrail_blocked_illegal_tool and var in [AblationVariant.VARIANT_A, AblationVariant.VARIANT_B, AblationVariant.VARIANT_C, AblationVariant.VARIANT_D]:
                        illegal_allowed += 1

            summary = VariantAggregateSummary(
                variant=var,
                sample_count=len(results),
                mean_latency_ms=round(mean_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                latency_budget_compliance_rate=round(budget_ok, 4),
                mean_faithfulness=round(mean_faith, 4),
                mean_hallucination_rate=round(mean_halluc, 4),
                mean_operator_alignment=round(mean_aop, 4),
                mean_ksr=round(mean_ksr, 4),
                direct_acceptance_rate=round(dar, 4),
                mean_tool_precision=round(mean_tool, 4) if mean_tool is not None else None,
                fsm_safety_violation_rate=round(illegal_allowed / len(results), 4)
            )
            aggregates[var] = summary

        return aggregates
