"""
Semantic Edit-Delta Extractor for HITL Customer Support.
Captures structural, lexical, and semantic divergences between
the raw AI suggestion (S_gen) and final human operator modifications (S_human).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
import difflib
import re
from pydantic import BaseModel, Field

from src.evaluation.alignment import calculate_ksr, calculate_semantic_similarity


class EditCategory(str, Enum):
    NO_EDIT = "NO_EDIT"                       # S_human == S_gen
    FACTUAL_CORRECTION = "FACTUAL_CORRECTION" # Numbers, policies, dates or key terms changed
    EXPANSION_DETAIL = "EXPANSION_DETAIL"     # Human added missing guidelines, steps or warnings
    PRUNING_DELETION = "PRUNING_DELETION"     # Human removed unnecessary or hallucinatory claims
    STYLE_REFORMULATION = "STYLE_REFORMULATION" # Tone, empathy or grammatical rewording


class SemanticDelta(BaseModel):
    ksr: float = Field(description="Keystroke Savings Rate (1.0 = identical)")
    semantic_similarity: float = Field(description="Cosine or token similarity")
    edit_category: EditCategory = Field(description="Categorization of the edit type")
    is_significant: bool = Field(description="True if human intervention changed meaning or content")
    added_spans: List[str] = Field(default_factory=list, description="Text added by operator")
    removed_spans: List[str] = Field(default_factory=list, description="Text removed by operator")
    diff_summary: str = Field(description="Human-readable summary of the adaptation delta")


class EditDeltaExtractor:
    """
    Extracts structured semantic edit deltas from operator interactions.
    """

    def __init__(self, ksr_significance_threshold: float = 0.95):
        self.ksr_threshold = ksr_significance_threshold

    def extract(
        self,
        s_gen: str,
        s_human: str,
        query: str = "",
        embedder: Optional[Any] = None
    ) -> SemanticDelta:
        s_gen_clean = s_gen.strip()
        s_human_clean = s_human.strip()

        if s_gen_clean == s_human_clean:
            return SemanticDelta(
                ksr=1.0,
                semantic_similarity=1.0,
                edit_category=EditCategory.NO_EDIT,
                is_significant=False,
                added_spans=[],
                removed_spans=[],
                diff_summary="Sugerencia aceptada íntegramente sin modificaciones."
            )

        ksr = calculate_ksr(s_gen_clean, s_human_clean)
        sim_sem = calculate_semantic_similarity(s_gen_clean, s_human_clean, embedder=embedder)

        # Token diff extraction using SequenceMatcher
        gen_words = re.findall(r'\S+|\s+', s_gen_clean)
        human_words = re.findall(r'\S+|\s+', s_human_clean)

        matcher = difflib.SequenceMatcher(None, gen_words, human_words)
        added_spans: List[str] = []
        removed_spans: List[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                removed_text = "".join(gen_words[i1:i2]).strip()
                added_text = "".join(human_words[j1:j2]).strip()
                if removed_text:
                    removed_spans.append(removed_text)
                if added_text:
                    added_spans.append(added_text)
            elif tag == 'delete':
                removed_text = "".join(gen_words[i1:i2]).strip()
                if removed_text:
                    removed_spans.append(removed_text)
            elif tag == 'insert':
                added_text = "".join(human_words[j1:j2]).strip()
                if added_text:
                    added_spans.append(added_text)

        len_gen = len(s_gen_clean)
        len_human = len(s_human_clean)

        # Look for numbers/policies/dates swapped
        digits_pattern = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
        added_digits = digits_pattern.findall(" ".join(added_spans))
        removed_digits = digits_pattern.findall(" ".join(removed_spans))
        is_number_replacement = bool(added_digits and removed_digits and added_digits != removed_digits)

        if len_human < len_gen * 0.75 and len(removed_spans) > 0:
            category = EditCategory.PRUNING_DELETION
            summary = f"Depuración/recorte: El operador eliminó afirmaciones innecesarias o redundantes: '{', '.join(removed_spans[:2])}'."
        elif len_human > len_gen * 1.25 and len(added_spans) > 0:
            category = EditCategory.EXPANSION_DETAIL
            summary = f"Ampliación de detalles: El operador añadió especificaciones: '{', '.join(added_spans[:2])}'."
        elif is_number_replacement:
            category = EditCategory.FACTUAL_CORRECTION
            summary = f"Corrección factual/numérica: Se sustituyó '{', '.join(removed_spans[:2])}' por '{', '.join(added_spans[:2])}'."
        elif (sim_sem < 0.70 and len(removed_spans) > 0 and len(added_spans) > 0):
            category = EditCategory.FACTUAL_CORRECTION
            summary = f"Corrección de contenido: Se sustituyó '{', '.join(removed_spans[:2])}' por '{', '.join(added_spans[:2])}'."
        else:
            category = EditCategory.STYLE_REFORMULATION
            summary = "Reformulación de estilo, tono o cortesía en la respuesta."

        is_significant = (
            ksr < self.ksr_threshold or
            category in [EditCategory.FACTUAL_CORRECTION, EditCategory.EXPANSION_DETAIL, EditCategory.PRUNING_DELETION]
        )

        return SemanticDelta(
            ksr=round(ksr, 4),
            semantic_similarity=round(sim_sem, 4),
            edit_category=category,
            is_significant=is_significant,
            added_spans=added_spans,
            removed_spans=removed_spans,
            diff_summary=summary
        )
