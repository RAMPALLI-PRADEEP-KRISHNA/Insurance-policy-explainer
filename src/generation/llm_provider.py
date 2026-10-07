from typing import Protocol, Optional

class LLMProvider(Protocol):
    """Protocol defining the required interface for any LLM provider in the RAG pipeline."""

    def validate_model(self) -> None:
        """
        Verify that the configured model identifier is recognized and available.
        Should raise ModelUnavailableError if the model is invalid.
        """
        ...

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[type] = None,
    ) -> str:
        """
        Generate grounded text using the underlying LLM.

        Args:
            prompt: Grounded user prompt including evidence and question.
            system_instruction: Strict grounding instructions.
            temperature: Sampling temperature.
            response_schema: Optional Pydantic model for structured JSON output.

        Returns:
            Generated text content string.
        """
        ...
