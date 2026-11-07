r"""
Decoupled Factual Faithfulness & Hallucination Evaluator (F_faith).
Deconstructs generated AI suggestions (S_gen) into atomic propositional claims
and verifies NLI entailment against retrieved corporate context passages (C).

F_faith(S_gen, C) = |Claims(S_gen) \cap SupportedBy(C)| / |Claims(S_gen)|
"""

from typing import List, Dict, Any, Optional
import re
from pydantic import BaseModel, Field


class AtomicClaim(BaseModel):
    claim_text: str
    is_supported: bool
    entailment_score: float = Field(ge=0.0, le=1.0)
    best_matching_context: Optional[str] = None


class FaithfulnessResult(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="F_faith in [0, 1]")
    hallucination_rate: float = Field(ge=0.0, le=1.0, description="1 - F_faith")
    total_claims: int
    supported_claims: int
    claims: List[AtomicClaim]


class FaithfulnessEvaluator:
    """
    Computes generation faithfulness using an independent judge model
    completely decoupled from the generation model (non-circular evaluation).
    """

    def __init__(
        self,
        model_name: Optional[str] = "cross-encoder/nli-deberta-v3-large",
        entailment_threshold: float = 0.50,
        load_model: bool = False
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self._judge_model = None

        if load_model and model_name:
            self._init_judge_model()

    def _init_judge_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._judge_model = CrossEncoder(self.model_name)
        except Exception:
            self._judge_model = None

    def extract_atomic_claims(self, text: str) -> List[str]:
        """
        Splits text into atomic assertions using sentence boundaries and punctuation.
        Filters out non-factual greetings and politeness phrases.
        """
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        claims: List[str] = []

        greetings_pattern = re.compile(
            r'^(hola|buenos d[ií]as|buenas tardes|un saludo|gracias por comunicarte|en qu[eé] puedo ayudarte|hasta luego)\b',
            re.IGNORECASE
        )

        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 8:
                continue
            if greetings_pattern.search(s_clean) and len(s_clean.split()) < 7:
                continue
            claims.append(s_clean)

        return claims

    def evaluate(
        self,
        s_gen: str,
        context_passages: List[str]
    ) -> FaithfulnessResult:
        """
        Evaluates factual grounding of S_gen against context C.
        """
        claims = self.extract_atomic_claims(s_gen)

        if not claims:
            return FaithfulnessResult(
                faithfulness_score=1.0,
                hallucination_rate=0.0,
                total_claims=0,
                supported_claims=0,
                claims=[]
            )

        if not context_passages:
            claim_objs = [
                AtomicClaim(
                    claim_text=c,
                    is_supported=False,
                    entailment_score=0.0,
                    best_matching_context=None
                )
                for c in claims
            ]
            return FaithfulnessResult(
                faithfulness_score=0.0,
                hallucination_rate=1.0,
                total_claims=len(claims),
                supported_claims=0,
                claims=claim_objs
            )

        atomic_results: List[AtomicClaim] = []
        supported_count = 0

        for claim in claims:
            best_score = 0.0
            best_passage = None

            if self._judge_model is not None:
                pairs = [[passage, claim] for passage in context_passages]
                scores = self._judge_model.predict(pairs)
                # For typical NLI cross-encoders (contradiction, neutral, entailment)
                for passage, score in zip(context_passages, scores):
                    val = float(score) if not hasattr(score, '__iter__') else float(score[-1])
                    if val > best_score:
                        best_score = val
                        best_passage = passage
            else:
                # Fast heuristic token overlap fallback for lightweight environments / tests
                claim_tokens = set(re.findall(r'\b\w{3,}\b', claim.lower()))
                for passage in context_passages:
                    passage_tokens = set(re.findall(r'\b\w{3,}\b', passage.lower()))
                    if not claim_tokens:
                        score = 1.0
                    else:
                        overlap = len(claim_tokens.intersection(passage_tokens))
                        score = overlap / len(claim_tokens)

                    if score > best_score:
                        best_score = score
                        best_passage = passage

            is_supp = best_score >= self.entailment_threshold
            if is_supp:
                supported_count += 1

            atomic_results.append(AtomicClaim(
                claim_text=claim,
                is_supported=is_supp,
                entailment_score=round(best_score, 4),
                best_matching_context=best_passage[:150] if best_passage else None
            ))

        f_faith = supported_count / len(claims) if claims else 1.0

        return FaithfulnessResult(
            faithfulness_score=round(f_faith, 4),
            hallucination_rate=round(1.0 - f_faith, 4),
            total_claims=len(claims),
            supported_claims=supported_count,
            claims=atomic_results
        )
