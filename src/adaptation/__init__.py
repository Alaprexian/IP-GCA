"""
IP-GCA Online Adaptation Module.
Captures semantic edit deltas (Delta_edit) from human operator modifications (S_human),
maintains dynamic attention preference keys (k_pref) with exponential temporal decay (gamma),
conditions generation prompts in real time, and triggers asynchronous LoRA updates.
"""

from src.adaptation.edit_extractor import (
    EditCategory,
    SemanticDelta,
    EditDeltaExtractor
)
from src.adaptation.preference_memory import (
    PreferenceKey,
    PreferenceMemory
)
from src.adaptation.conditioning_bridge import (
    EditDeltaConditioningBridge
)
from src.adaptation.lora_trigger import (
    LoRABatchDataset,
    LoRATriggerController
)

__all__ = [
    "EditCategory",
    "SemanticDelta",
    "EditDeltaExtractor",
    "PreferenceKey",
    "PreferenceMemory",
    "EditDeltaConditioningBridge",
    "LoRABatchDataset",
    "LoRATriggerController"
]
