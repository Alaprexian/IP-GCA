"""
Unit tests for the Local LLMGenerator module.
Tests device mapping, lazy loading, chat templating, and prompt formatting.
"""

import pytest
import torch
from pathlib import Path
from src.copilot.generator import LLMGenerator
from src.copilot.orchestrator import IPGCAOrchestrator


def test_generator_lazy_loading():
    # Verify that lazy_load=True does not load weights immediately
    gen = LLMGenerator(
        model_name_or_path="models/Qwen2.5-1.5B-Instruct",
        load_in_4bit=True,
        lazy_load=True
    )
    assert not gen.is_loaded
    assert gen.model is None


@pytest.mark.skipif(not torch.cuda.is_available() or not Path("models/Qwen2.5-1.5B-Instruct").exists(),
                    reason="Requires CUDA and downloaded local Qwen2.5-1.5B model")
def test_generator_real_inference():
    gen = LLMGenerator(
        model_name_or_path="models/Qwen2.5-1.5B-Instruct",
        load_in_4bit=True,
        lazy_load=False
    )
    assert gen.is_loaded
    assert gen.model is not None

    response = gen.generate(
        prompt="¿Cuál es la capital de Francia?",
        system_prompt="Responde en una sola palabra.",
        max_new_tokens=10,
        temperature=0.0
    )
    assert "París" in response or "Paris" in response


def test_orchestrator_with_generator_flag():
    # Orchestrator with enable_real_llm=False stays fast
    orch = IPGCAOrchestrator(
        enable_reranker=False,
        enable_real_llm=False
    )
    assert orch.generator is None

    res = orch.process_turn(
        query="¿Cuánto tiempo para devoluciones?",
        intent_probs={"refund_request": 0.9}
    )
    assert len(res.s_gen) > 5
