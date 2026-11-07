"""
IP-GCA Copilot Module.
Provides the central orchestrator coordinating multi-turn customer support interactions.
"""

from src.copilot.orchestrator import (
    IPGCAOrchestrator,
    CopilotTurnResponse
)
from src.copilot.generator import LLMGenerator

__all__ = [
    "IPGCAOrchestrator",
    "CopilotTurnResponse",
    "LLMGenerator"
]

