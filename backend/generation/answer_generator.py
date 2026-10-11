"""Answer generator module for grounded insurance policy Q&A."""

import json
from typing import Optional
from backend.config import AppConfig, config as default_config
from backend.generation.citation_assembler import assemble_citations
from backend.generation.llm_provider import LLMProvider
from backend.generation.gemini_client import GeminiClient, gemini_client as default_gemini_client
from backend.models.data_models import Evidence, EvidenceStatus, GroundedAnswer
from backend.utils.logging_utils import get_logger
from backend.utils.exceptions import (
    MistralRateLimitError,
    GeminiRateLimitError,
    GenerationTimeoutError,
    GeminiServiceUnavailableError,
    MistralServiceUnavailableError,
    MistralAuthError,
    MistralTierError,
    ModelUnavailableError,
    MissingAPIKeyError,
    GenerationError,
)

logger = get_logger(__name__)

# Strict Grounded System Prompt enforcing all 7 safety rules
GROUNDED_SYSTEM_INSTRUCTION = """You are an expert insurance policy assistant. Your sole purpose is to provide clear, accurate, plain-language answers grounded strictly in the provided policy excerpts.

You MUST obey the following 7 inviolable safety rules:
1. EVIDENCE ONLY: Answer exclusively using the policy text explicitly supplied in the Evidence section.
2. NO OUTSIDE KNOWLEDGE: Do not rely on general insurance knowledge or external assumptions to fill in missing details.
3. NO GUESSING: If the evidence does not clearly provide the answer, explicitly state that the evidence does not specify it.
4. PRESERVE NUMBERS & LIMITS VERBATIM: Never alter, recalculate, or approximate any numbers, percentages (e.g. 10%, 20%), currency amounts (e.g. ₹5,00,000, Rs. 5,000, 5 lakh), durations (e.g. 36 months, 24 months, 30 days), or ages (e.g. age 61 and above). For example, '36 months' must NEVER be changed to '3 years'.
5. PRESERVE NEGATION & EXCLUSIONS: Never convert negative coverage statements into positive ones (e.g., 'not covered' must never become 'covered'; 'excluded' must never become 'included').
6. PRESERVE CONDITIONAL QUALIFIERS: If evidence says '10% co-pay for age 61 and above', the age condition must not be omitted. Always preserve all age, network, disease, or procedural conditions.
7. DO NOT OVERSTATE: Never use absolute terms like 'always', 'never', 'guaranteed', or 'fully covered' unless explicitly stated in the evidence.
"""

def get_llm_provider(config: AppConfig) -> LLMProvider:
    provider = config.LLM_PROVIDER
    if provider == "fake":
        return default_gemini_client
    from backend.generation.provider_chain import ProviderChain
    return ProviderChain(config)

class AnswerGenerator:
    """
    Orchestrates grounded answer generation from validated Evidence.
    Enforces strict evidence-state behavior, local citation assembly, and plain English output.
    """

    def __init__(
        self,
        llm_provider_instance: Optional[LLMProvider] = None,
        gemini_client_instance: Optional[LLMProvider] = None,
        app_config: Optional[AppConfig] = None,
    ):
        self.config = app_config or default_config
        # Prefer llm_provider_instance, fallback to gemini_client_instance (for legacy tests), then get_llm_provider
        self.llm_provider = llm_provider_instance or gemini_client_instance or get_llm_provider(self.config)

    def _generate_grounded_text(self, query: str, evidence: Evidence, language: str = "en") -> str:
        """Generate strictly grounded answer text via LLM provider."""
        prompt = (
            f"Question:\n{query}\n\n"
            f"Evidence Excerpts from Policy:\n{evidence.evidence_summary or ''}\n\n"
            f"Instruction: Provide a concise, plain-language answer following all 7 grounding rules."
        )

        return self.llm_provider.generate(
            prompt=prompt,
            system_instruction=GROUNDED_SYSTEM_INSTRUCTION,
        )

    def generate_answer(
        self,
        query: str,
        evidence: Evidence,
        language: str = "en",
    ) -> GroundedAnswer:
        """
        Generate a structured, fully grounded answer with authoritative citations.

        Args:
            query: User's original question.
            evidence: Validated Evidence object.
            language: Target output language (defaults to 'en').

        Returns:
            GroundedAnswer with answer text, citations, and evidence provenance.
        """
        clean_lang = language.strip().lower() if language else "en"
        citations = assemble_citations(evidence)

        # State 1: INSUFFICIENT — Controlled Abstention without calling LLM
        if evidence.status == EvidenceStatus.INSUFFICIENT:
            refusal_text = (
                "I could not find sufficient information in the uploaded policy to answer this question. "
                "Please consult the complete policy document or contact your insurer for clarification."
            )

            return GroundedAnswer(
                query=query,
                answer=refusal_text,
                language=clean_lang,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                citations=[],
                supporting_chunk_ids=[],
                confidence=evidence.confidence,
                detected_conflicts=[],
                generation_model=str(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None))) if isinstance(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None)), str) else None,
            )

        # State 2: CONFLICTING — Explicit Contradiction Report without picking a side
        if evidence.status == EvidenceStatus.CONFLICTING:
            conflict_details = "\n".join(f"- {c}" for c in evidence.detected_conflicts)
            conflict_text = (
                f"The retrieved policy sections contain conflicting information:\n{conflict_details}\n\n"
                f"Please verify with your insurer or refer to the primary policy schedule before relying on either figure."
            )

            return GroundedAnswer(
                query=query,
                answer=conflict_text,
                language=clean_lang,
                evidence_status=EvidenceStatus.CONFLICTING,
                citations=citations,
                supporting_chunk_ids=evidence.supporting_chunk_ids,
                confidence=evidence.confidence,
                detected_conflicts=evidence.detected_conflicts,
                generation_model=str(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None))) if isinstance(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None)), str) else None,
            )

        # State 3: MARGINAL — Cautious Answer without Guessing
        if evidence.status == EvidenceStatus.MARGINAL:
            summary_snippet = evidence.evidence_summary or ""
            marginal_text = (
                f"The retrieved policy sections discuss related terms, but do not clearly or definitively specify "
                f"the exact condition asked in your question.\n\nRelevant policy excerpt:\n{summary_snippet}"
            )

            return GroundedAnswer(
                query=query,
                answer=marginal_text,
                language=clean_lang,
                evidence_status=EvidenceStatus.MARGINAL,
                citations=citations,
                supporting_chunk_ids=evidence.supporting_chunk_ids,
                confidence=evidence.confidence,
                detected_conflicts=[],
                generation_model=str(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None))) if isinstance(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None)), str) else None,
            )

                # State 4: SUFFICIENT - Direct Grounded Answer via Gemini
        try:
            answer_text = self._generate_grounded_text(query, evidence, language=clean_lang)
            return GroundedAnswer(
                query=query,
                answer=answer_text,
                language=clean_lang,
                evidence_status=EvidenceStatus.SUFFICIENT,
                citations=citations,
                supporting_chunk_ids=evidence.supporting_chunk_ids,
                confidence=evidence.confidence,
                detected_conflicts=[],
                generation_model=str(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", self.config.GEMINI_MODEL))) if isinstance(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", self.config.GEMINI_MODEL)), str) else self.config.GEMINI_MODEL,
            )
        except MissingAPIKeyError:
            raise
        except (
            GenerationTimeoutError,
            GeminiServiceUnavailableError,
            MistralServiceUnavailableError,
            MistralAuthError,
            MistralTierError,
            MistralRateLimitError,
            GeminiRateLimitError,
            ModelUnavailableError,
            GenerationError,
        ) as e:
            return GroundedAnswer(
                query=query,
                answer=evidence.evidence_summary or " ",
                language=clean_lang,
                evidence_status=EvidenceStatus.SUFFICIENT,
                citations=citations,
                supporting_chunk_ids=evidence.supporting_chunk_ids,
                confidence=evidence.confidence,
                detected_conflicts=[],
                generation_model=str(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None))) if isinstance(getattr(self.llm_provider, "last_used_model", getattr(self.llm_provider, "model", None)), str) else None,
                llm_unavailable=True,
                llm_unavailable_reason=str(e),
            )

answer_generator = AnswerGenerator()



