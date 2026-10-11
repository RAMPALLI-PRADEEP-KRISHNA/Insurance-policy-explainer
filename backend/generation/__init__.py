"""Generation package for grounded answer synthesis and citation assembly."""

from backend.generation.gemini_client import GeminiClient, gemini_client
from backend.generation.citation_assembler import assemble_citations
from backend.generation.answer_generator import (
    AnswerGenerator,
    answer_generator,
    GROUNDED_SYSTEM_INSTRUCTION,
)

__all__ = [
    "GeminiClient",
    "gemini_client",
    "assemble_citations",
    "AnswerGenerator",
    "answer_generator",
    "GROUNDED_SYSTEM_INSTRUCTION",
]
