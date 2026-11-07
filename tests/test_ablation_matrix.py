"""
Unit tests for the IP-GCA Ablation Matrix (Variants A–H).
Validates variant pipeline behaviors, metric generation, and reporting.
"""

import pytest
from src.evaluation.ablation_matrix import (
    AblationMatrixRunner,
    AblationVariant,
    VariantSampleResult,
    VariantAggregateSummary
)
from src.evaluation.benchmark_dataset import (
    BenchmarkSample,
    get_default_ablation_benchmark_dataset
)


@pytest.fixture(scope="module")
def ablation_runner():
    return AblationMatrixRunner(fast_dev_mode=True)


def test_ablation_sample_run_variant_a(ablation_runner):
    sample = BenchmarkSample(
        sample_id="T_01",
        query="¿Cuál es el plazo de reembolso?",
        category="policy_faq",
        intent_probs={"refund_request": 0.90}
    )
    res: VariantSampleResult = ablation_runner.run_sample_on_variant(
        sample=sample,
        variant=AblationVariant.VARIANT_A
    )
    assert res.variant == AblationVariant.VARIANT_A
    assert res.selected_path == "FIXED_PIPELINE"
    assert res.latency_ms > 0
    assert 0.0 <= res.faithfulness <= 1.0


def test_ablation_sample_run_variant_e_full_ipgca(ablation_runner):
    sample = BenchmarkSample(
        sample_id="T_02",
        query="¿Cuánto tiempo tengo para devolver el producto?",
        category="policy_faq",
        intent_probs={"refund_request": 0.85, "support": 0.15}
    )
    res: VariantSampleResult = ablation_runner.run_sample_on_variant(
        sample=sample,
        variant=AblationVariant.VARIANT_E
    )
    assert res.variant == AblationVariant.VARIANT_E
    assert res.selected_path in ["FAST_PARAMETRIC", "HYBRID_RAG", "AGENTIC_REFLECTION", "ASSISTED_ABSTENTION"]
    assert res.latency_ms > 0.0


def test_ablation_variant_h_randomized_context_degradation(ablation_runner):
    # In Variant H, context is replaced with unrelated distractors, so faithfulness should drop
    sample = BenchmarkSample(
        sample_id="T_03",
        query="¿Cuál es la política de garantías corporativas para laptops?",
        category="policy_faq",
        intent_probs={"warranty": 0.90}
    )
    res_e = ablation_runner.run_sample_on_variant(sample, AblationVariant.VARIANT_E)
    res_h = ablation_runner.run_sample_on_variant(sample, AblationVariant.VARIANT_H)

    assert res_h.variant == AblationVariant.VARIANT_H
    # Variant H uses random context, so grounding/faithfulness should be lower than or equal to E
    assert res_h.faithfulness <= res_e.faithfulness or res_h.hallucination_rate >= res_e.hallucination_rate


def test_ablation_matrix_aggregate_summary(ablation_runner):
    dataset = get_default_ablation_benchmark_dataset()[:2] # Fast subset
    subset_variants = [AblationVariant.VARIANT_A, AblationVariant.VARIANT_E]

    aggregates = ablation_runner.run_benchmark_matrix(
        samples=dataset,
        variants_to_test=subset_variants
    )

    assert AblationVariant.VARIANT_A in aggregates
    assert AblationVariant.VARIANT_E in aggregates

    summary_a: VariantAggregateSummary = aggregates[AblationVariant.VARIANT_A]
    summary_e: VariantAggregateSummary = aggregates[AblationVariant.VARIANT_E]

    assert summary_a.sample_count == 2
    assert summary_e.sample_count == 2
    assert summary_a.mean_latency_ms > 0
    assert summary_e.mean_latency_ms > 0
