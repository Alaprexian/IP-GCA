"""
Dynamic Preference Memory with Exponential Temporal Decay.
Stores operator preference keys (k_pref) derived from semantic deltas (Delta_edit)
and weights them according to recency: w_i(t) = w_0 * gamma^(t - t_i).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import uuid
import json
from pathlib import Path
import numpy as np
import yaml
from pydantic import BaseModel, Field

from src.adaptation.edit_extractor import SemanticDelta, EditCategory


class PreferenceKey(BaseModel):
    key_id: str
    query: str
    intent: Optional[str] = None
    s_gen: str
    s_human: str
    delta: SemanticDelta
    turn_created: int
    current_weight: float = 1.0
    vector: Optional[List[float]] = None
    is_synced_lora: bool = False


class PreferenceMemory:
    """
    In-memory vector store of operator preference keys with dynamic temporal decay.
    """

    def __init__(
        self,
        config_path: str = "configs/config_default.yaml",
        min_weight_threshold: float = 0.05,
        max_capacity: int = 500
    ):
        self.decay_factor_gamma = 0.95
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                adapt_cfg = cfg.get("adaptation", {})
                self.decay_factor_gamma = float(adapt_cfg.get("decay_factor_gamma", 0.95))
        except Exception:
            pass

        self.min_weight_threshold = min_weight_threshold
        self.max_capacity = max_capacity
        self.current_turn: int = 0
        self.keys: Dict[str, PreferenceKey] = {}

    def step_turn(self) -> None:
        """
        Advance global session turn counter and recalculate decayed weights:
        w_i(t) = gamma^(t - t_i)
        """
        self.current_turn += 1
        keys_to_remove = []

        for key_id, pref in self.keys.items():
            delta_t = self.current_turn - pref.turn_created
            new_weight = math.pow(self.decay_factor_gamma, delta_t)
            pref.current_weight = round(new_weight, 5)

            # Evict decayed keys unless they are waiting for LoRA synchronization
            if pref.current_weight < self.min_weight_threshold and pref.is_synced_lora:
                keys_to_remove.append(key_id)

        for k in keys_to_remove:
            del self.keys[k]

    def add_preference(
        self,
        query: str,
        s_gen: str,
        s_human: str,
        delta: SemanticDelta,
        intent: Optional[str] = None,
        embedder: Optional[Any] = None
    ) -> Optional[PreferenceKey]:
        """
        Registers an operator edit delta as a preference key (k_pref).
        Ignores non-significant deltas (e.g. perfect acceptance).
        """
        if not delta.is_significant or delta.edit_category == EditCategory.NO_EDIT:
            return None

        key_id = f"pref_{uuid.uuid4().hex[:8]}"

        vector = None
        if embedder is not None:
            text_to_embed = f"{query} | {delta.diff_summary} | {' '.join(delta.added_spans)}"
            raw_emb = embedder.encode(text_to_embed)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                vector = (raw_emb / norm).tolist()

        pref = PreferenceKey(
            key_id=key_id,
            query=query,
            intent=intent,
            s_gen=s_gen,
            s_human=s_human,
            delta=delta,
            turn_created=self.current_turn,
            current_weight=1.0,
            vector=vector,
            is_synced_lora=False
        )

        if len(self.keys) >= self.max_capacity:
            lowest_key = min(self.keys.values(), key=lambda x: x.current_weight)
            del self.keys[lowest_key.key_id]

        self.keys[key_id] = pref
        return pref

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 3,
        embedder: Optional[Any] = None,
        min_weight: float = 0.10
    ) -> List[Tuple[PreferenceKey, float]]:
        """
        Retrieves top-k preference keys conditioned on query similarity and temporal decay weight:
        score = similarity(query, k_pref) * w_i(t)
        """
        if not self.keys:
            return []

        active_keys = [k for k in self.keys.values() if k.current_weight >= min_weight]
        if not active_keys:
            return []

        scored: List[Tuple[PreferenceKey, float]] = []

        if embedder is not None:
            query_emb = embedder.encode(query)
            q_norm = np.linalg.norm(query_emb)
            if q_norm > 0:
                query_emb = query_emb / q_norm

            for pref in active_keys:
                if pref.vector is not None:
                    pref_vec = np.array(pref.vector)
                    cos_sim = float(np.dot(query_emb, pref_vec))
                else:
                    cos_sim = self._lexical_overlap(query, pref.query)

                composite_score = max(0.0, cos_sim) * pref.current_weight
                scored.append((pref, composite_score))
        else:
            for pref in active_keys:
                overlap = self._lexical_overlap(query, pref.query)
                composite_score = overlap * pref.current_weight
                scored.append((pref, composite_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _lexical_overlap(text_a: str, text_b: str) -> float:
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        union = set_a.union(set_b)
        if not union:
            return 0.0
        return len(set_a.intersection(set_b)) / len(union)

    def get_unprocessed_batch(self, limit: int = 32) -> List[PreferenceKey]:
        """Returns unsynced deltas for batch LoRA micro-training."""
        return [k for k in self.keys.values() if not k.is_synced_lora][:limit]

    def mark_batch_synced(self, key_ids: List[str]) -> None:
        """Marks specified deltas as synced with offline LoRA."""
        for kid in key_ids:
            if kid in self.keys:
                self.keys[kid].is_synced_lora = True

    def save(self, path: str) -> None:
        """
        Persists the current preference memory state to a JSON file.
        Allows preferences to survive process restarts between sessions.
        """
        out = {
            "current_turn": self.current_turn,
            "decay_factor_gamma": self.decay_factor_gamma,
            "keys": {k: v.model_dump() for k, v in self.keys.items()}
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        Restores preference memory state from a previously saved JSON file.
        Skips gracefully if the file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            return
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.current_turn = data.get("current_turn", 0)
        self.decay_factor_gamma = data.get("decay_factor_gamma", self.decay_factor_gamma)
        raw_keys = data.get("keys", {})
        self.keys = {
            k: PreferenceKey.model_validate(v) for k, v in raw_keys.items()
        }

    def clear(self) -> None:
        self.keys.clear()
        self.current_turn = 0
