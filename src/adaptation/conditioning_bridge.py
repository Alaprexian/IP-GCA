"""
In-Context Conditioning Bridge for Operator Preferences.
Translates active preference keys (k_pref) into dynamic prompt constraints
and soft attention guidance injected into the generator LLM.
"""

from typing import List, Tuple, Optional, Dict, Any
from src.adaptation.preference_memory import PreferenceKey


class EditDeltaConditioningBridge:
    """
    Transforms retrieved operator preference keys into structured in-context constraints.
    """

    def __init__(self, max_preferences_in_prompt: int = 3):
        self.max_preferences = max_preferences_in_prompt

    def format_preference_constraints(
        self,
        ranked_preferences: List[Tuple[PreferenceKey, float]],
        min_score_threshold: float = 0.15
    ) -> str:
        """
        Builds a dedicated prompt section containing high-confidence operator corrections.
        """
        filtered = [
            (pref, score) for pref, score in ranked_preferences
            if score >= min_score_threshold
        ][:self.max_preferences]

        if not filtered:
            return ""

        lines = [
            "### REGLAS DE ALINEACIÓN DEL OPERADOR HUMANO (HISTORIAL HITL RECIENTE):",
            "Ten en cuenta las siguientes preferencias y correcciones efectuadas por supervisores humanos para casos similares:"
        ]

        for i, (pref, score) in enumerate(filtered, start=1):
            category_label = pref.delta.edit_category.value
            lines.append(f"{i}. [Precedente (Relevancia: {score:.2f}, Tipo: {category_label})]")
            lines.append(f"   - Contexto similar: \"{pref.query}\"")
            lines.append(f"   - Directiva aprendida: {pref.delta.diff_summary}")
            
            if pref.delta.added_spans:
                added_str = ", ".join(f'"{s}"' for s in pref.delta.added_spans[:3])
                lines.append(f"   - Debes incluir/priorizar: {added_str}")
                
            if pref.delta.removed_spans:
                removed_str = ", ".join(f'"{s}"' for s in pref.delta.removed_spans[:3])
                lines.append(f"   - Evitar afirmar: {removed_str}")

        lines.append("Aplica estas directivas de forma estricta sin contradecir las políticas generales.")
        return "\n".join(lines)

    def assemble_generation_prompt(
        self,
        user_query: str,
        retrieved_passages: List[str],
        ranked_preferences: List[Tuple[PreferenceKey, float]],
        base_system_prompt: Optional[str] = None
    ) -> str:
        """
        Assembles the complete grounded & preference-conditioned prompt for the LLM generator.
        """
        sys_prompt = base_system_prompt or (
            "Eres el asistente inteligente de atención al cliente de la empresa (IP-GCA Copilot). "
            "Responde de manera precisa, veraz, profesional y fundamentada exclusivamente en la información oficial provista."
        )

        context_str = ""
        if retrieved_passages:
            context_str = "### CONTEXTO DE CONOCIMIENTO EMPRESARIAL RECUPERADO:\n" + "\n---\n".join(retrieved_passages)

        pref_str = self.format_preference_constraints(ranked_preferences)

        sections = [sys_prompt]
        if context_str:
            sections.append(context_str)
        if pref_str:
            sections.append(pref_str)

        sections.append(f"### CONSULTA DEL CLIENTE:\n{user_query}\n\n### RESPUESTA SUGERIDA:")

        return "\n\n".join(sections)

    def assemble_chat_messages(
        self,
        user_query: str,
        retrieved_passages: List[str],
        ranked_preferences: List[Tuple[PreferenceKey, float]],
        base_system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Assembles structured chat messages (system and user) for chat-templated LLMs.
        """
        sys_prompt = base_system_prompt or (
            "Eres el asistente inteligente de atención al cliente de la empresa (IP-GCA Copilot). "
            "Responde de manera precisa, veraz, profesional y fundamentada exclusivamente en la información oficial provista. "
            "Sé conciso y directo en tu respuesta."
        )

        system_parts = [sys_prompt]
        if retrieved_passages:
            context_str = "### CONTEXTO DE CONOCIMIENTO EMPRESARIAL RECUPERADO:\n" + "\n---\n".join(retrieved_passages)
            system_parts.append(context_str)

        pref_str = self.format_preference_constraints(ranked_preferences)
        if pref_str:
            system_parts.append(pref_str)

        full_system_content = "\n\n".join(system_parts)

        return [
            {"role": "system", "content": full_system_content},
            {"role": "user", "content": user_query}
        ]

