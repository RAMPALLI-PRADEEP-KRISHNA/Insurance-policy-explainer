"""Answer generator module for grounded insurance policy Q&A."""

import json
from typing import Optional
from src.config import AppConfig, config as default_config
from src.generation.citation_assembler import assemble_citations
from src.generation.llm_provider import LLMProvider
from src.generation.gemini_client import GeminiClient, gemini_client as default_gemini_client
from src.models.data_models import Evidence, EvidenceStatus, GroundedAnswer
from src.utils.logging_utils import get_logger
from src.utils.exceptions import MistralRateLimitError, GeminiRateLimitError, GenerationTimeoutError, GeminiServiceUnavailableError, MistralServiceUnavailableError, MistralAuthError, MissingAPIKeyError

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

TAMIL_TRANSLATION_INSTRUCTION = """You are a professional English-to-Tamil translator specializing in legal and insurance documents.
Translate the following grounded insurance answer into clear, natural, professional Tamil.

CRITICAL INVARIANTS:
1. Preserve all numbers, percentages, durations (e.g. 36 months, 30 days, 24 months), currency figures (e.g. Rs. 5,00,000, ₹5,000), and age limits (e.g. age 61) exactly as written.
2. Preserve all negations (e.g. excluded / not covered) and conditions verbatim.
3. Do NOT add new claims, legal advice, or general knowledge.
"""

def get_llm_provider(config: AppConfig) -> LLMProvider:
    provider = config.LLM_PROVIDER
    if provider == "fake":
        return default_gemini_client
    from src.generation.provider_chain import ProviderChain
    return ProviderChain(config)

class AnswerGenerator:
    """
    Orchestrates grounded answer generation from validated Evidence.
    Enforces strict evidence-state behavior, local citation assembly, and multilingual output.
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
        """Generate strictly grounded answer text via Gemini."""
        prompt = (
            f"Question:\n{query}\n\n"
            f"Evidence Excerpts from Policy:\n{evidence.evidence_summary or ''}\n\n"
            f"Instruction: Provide a concise, plain-language answer following all 7 grounding rules."
        )

        # English or direct generation
        if language == "ta" and self.config.TAMIL_GENERATION_MODE == "direct":
            tamil_direct_prompt = (
                f"கேள்வி (Question):\n{query}\n\n"
                f"பாலிசி சான்றுகள் (Policy Evidence Excerpts):\n{evidence.evidence_summary or ''}\n\n"
                f"அறிவுறுத்தல்: மேலே உள்ள பாலிசி சான்றுகளின் அடிப்படையில் மட்டுமே தமிழில் தெளிவான, சுருக்கமான பதிலை வழங்கவும். "
                f"எண்கள், சதவீதங்கள், கால அளவுகள் (36 months, 24 months, 10%), மற்றும் நிபந்தனைகளை துல்லியமாகப் பாதுகாக்கவும்."
            )
            return self.llm_provider.generate(
                prompt=tamil_direct_prompt,
                system_instruction=GROUNDED_SYSTEM_INSTRUCTION,
            )

        # Generate English grounded text
        english_answer = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=GROUNDED_SYSTEM_INSTRUCTION,
        )

        # If Tamil requested in translation mode
        if language == "ta":
            tamil_prompt = f"English Answer:\n{english_answer}\n\nTranslate to Tamil:"
            return self.llm_provider.generate(
                prompt=tamil_prompt,
                system_instruction=TAMIL_TRANSLATION_INSTRUCTION,
            )

        return english_answer

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
            evidence: Validated Evidence object from Phase 3.
            language: Target output language ('en' for English, 'ta' for Tamil).

        Returns:
            GroundedAnswer with answer text, citations, and evidence provenance.
        """
        clean_lang = language.strip().lower() if language else "en"
        citations = assemble_citations(evidence)

        # State 1: INSUFFICIENT — Controlled Abstention without calling LLM
        if evidence.status == EvidenceStatus.INSUFFICIENT:
            if clean_lang == "ta":
                refusal_text = (
                    "பதிவேற்றப்பட்ட பாலிசியில் இந்தக் கேள்விக்கு பதிலளிக்க போதுமான தகவல் கிடைக்கவில்லை. "
                    "முழுமையான பாலிசி ஆவணத்தைப் பார்க்கவும் அல்லது உங்கள் காப்பீட்டாளரைத் தொடர்பு கொள்ளவும்."
                )
            else:
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
            if clean_lang == "ta":
                conflict_text = (
                    f"பாலிசியின் பிரிவுகளில் முரண்பாடான தகவல்கள் கண்டறியப்பட்டுள்ளன:\n{conflict_details}\n\n"
                    f"எந்தவொரு தொகையையும் நம்புவதற்கு முன் பாலிசி விதிமுறைகளை சரிபார்க்கவும்."
                )
            else:
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
            if clean_lang == "ta":
                marginal_text = (
                    f"பாலிசியின் பிரிவுகள் தொடர்புடைய தலைப்பைப் பற்றி விவரிக்கின்றன, ஆனால் நீங்கள் கேட்ட குறிப்பிட்ட "
                    f"நிபந்தனையை தெளிவாகக் குறிப்பிடவில்லை.\n\nகிடைத்த சான்று:\n{summary_snippet}"
                )
            else:
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
        except (GenerationTimeoutError, GeminiServiceUnavailableError, MistralServiceUnavailableError, MistralAuthError, MistralRateLimitError, GeminiRateLimitError) as e:
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



