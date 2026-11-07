"""
Pytest configuration and shared fixtures for IP-GCA Copilot test suite.
Avoids redundant model downloading and re-initialization across test modules.
"""

import pytest
from pathlib import Path
from src.copilot.orchestrator import IPGCAOrchestrator
from src.guardrail.fsm import TransactionalGuardrailFSM
from src.retrieval.engine import HybridRetrievalEngine


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def default_config_path() -> str:
    return "configs/config_default.yaml"


@pytest.fixture(scope="session")
def default_fsm_rules_path() -> str:
    return "configs/fsm_rules.yaml"


@pytest.fixture(scope="session")
def default_kb_path() -> str:
    return "data/raw/corporate_knowledge_base.json"


@pytest.fixture(scope="session")
def shared_fast_orchestrator(default_config_path, default_fsm_rules_path, default_kb_path) -> IPGCAOrchestrator:
    """
    Session-scoped lightweight orchestrator with reranker and real LLM disabled.
    Saves ~30s of initialization overhead across tests.
    """
    return IPGCAOrchestrator(
        config_path=default_config_path,
        fsm_config_path=default_fsm_rules_path,
        kb_path=default_kb_path,
        enable_reranker=False,
        enable_real_llm=False
    )


@pytest.fixture
def fresh_fsm(default_fsm_rules_path) -> TransactionalGuardrailFSM:
    """
    Function-scoped fresh FSM instance reset to IDLE.
    """
    fsm = TransactionalGuardrailFSM(default_fsm_rules_path)
    fsm.reset()
    return fsm
