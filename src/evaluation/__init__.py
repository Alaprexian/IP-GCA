"""
IP-GCA Formal Evaluation Metrics.
Provides latency tracking, AST-based tool precision (A_tool),
operator alignment index (A_op, KSR, DAR), and decoupled faithfulness (F_faith).
"""

from src.evaluation.latency_tracker import LatencyTracker, RequestProfile, StageTimings
from src.evaluation.tool_precision import calculate_tool_precision
from src.evaluation.alignment import (
    calculate_ksr,
    calculate_semantic_similarity,
    calculate_operator_alignment,
    calculate_dar
)
from src.evaluation.faithfulness import (
    AtomicClaim,
    FaithfulnessResult,
    FaithfulnessEvaluator
)

__all__ = [
    "LatencyTracker",
    "RequestProfile",
    "StageTimings",
    "calculate_tool_precision",
    "calculate_ksr",
    "calculate_semantic_similarity",
    "calculate_operator_alignment",
    "calculate_dar",
    "AtomicClaim",
    "FaithfulnessResult",
    "FaithfulnessEvaluator",
    # NOTE: AblationMatrixRunner and BenchmarkSample are NOT auto-imported here
    # to avoid circular imports. Import them directly:
    #   from src.evaluation.ablation_matrix import AblationMatrixRunner, AblationVariant
    #   from src.evaluation.benchmark_dataset import get_default_ablation_benchmark_dataset
]
