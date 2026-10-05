"""Gemini API client wrapper using the modern Google GenAI SDK (google-genai)."""

import os
from typing import Optional, Any
from src.config import AppConfig, config as default_config
from src.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    MalformedOutputError,
    MissingAPIKeyError,
    ModelUnavailableError,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """
    Client wrapper for Google Gemini models using the modern `google-genai` SDK.
    Enforces strict model validation and robust error handling without leaking secrets.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        app_config: Optional[AppConfig] = None,
        client_instance: Optional[Any] = None,
    ):
        self.config = app_config or default_config
        self.api_key = api_key or self.config.GEMINI_API_KEY
        self.model_name = model_name or self.config.GEMINI_MODEL
        self._client = client_instance
        self._validated = False

    @property
    def client(self) -> Any:
        """Lazy-initialize the genai Client."""
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Gemini GenAI client: %s", e)
                raise GenerationError(f"Could not initialize GenAI client: {str(e)}") from e
        return self._client

    def validate_model(self) -> None:
        """
        Verify that the configured Gemini model identifier is recognized and available.
        Raises ModelUnavailableError if the model is not found or unsupported.
        """
        if self._validated:
            return

        # If running in mock / test environment with a custom client mock, mark validated
        if self._client is not None and not self.api_key:
            self._validated = True
            return

        logger.info("Validating Gemini model '%s'...", self.model_name)
        try:
            # Model identifiers in Google GenAI can be queried via client.models.get
            model_info = self.client.models.get(model=self.model_name)
            if not model_info:
                raise ModelUnavailableError(
                    f"Configured Gemini model '{self.model_name}' could not be verified."
                )
            self._validated = True
            logger.info("Gemini model '%s' successfully validated.", self.model_name)
        except MissingAPIKeyError:
            raise
        except Exception as e:
            err_str = str(e)
            logger.error("Model validation failed for '%s': %s", self.model_name, err_str)
            raise ModelUnavailableError(
                f"Configured Gemini model '{self.model_name}' is unavailable or invalid: {err_str}"
            ) from e

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[type] = None,
    ) -> str:
        """
        Generate content using the configured Gemini model.

        Args:
            prompt: Grounded user prompt including evidence and question.
            system_instruction: Strict grounding instructions.
            temperature: Sampling temperature (default: config.GEMINI_TEMPERATURE, 0.0 for deterministic grounding).
            response_schema: Optional Pydantic model for structured JSON output.

        Returns:
            Generated text content string.

        Raises:
            MissingAPIKeyError: If API key is missing.
            GenerationTimeoutError: If the call times out.
            MalformedOutputError: If the model returns an empty or invalid response.
            GenerationError: On general API or network failures.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        temp = temperature if temperature is not None else self.config.GEMINI_TEMPERATURE

        try:
            from google.genai import types

            # Build config
            config_params: dict = {
                "temperature": temp,
            }
            if system_instruction:
                config_params["system_instruction"] = system_instruction
            if response_schema:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_schema

            gen_config = types.GenerateContentConfig(**config_params)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=gen_config,
            )

            if not response or not response.text:
                raise MalformedOutputError(
                    f"Gemini model '{self.model_name}' returned an empty or malformed response."
                )

            return response.text.strip()

        except (MissingAPIKeyError, MalformedOutputError, GenerationTimeoutError):
            raise
        except TimeoutError as e:
            logger.error("Gemini generation timed out: %s", e)
            raise GenerationTimeoutError(f"Gemini generation timed out after {self.config.GEMINI_TIMEOUT_SECONDS}s.") from e
        except Exception as e:
            err_msg = str(e)
            logger.error("Gemini API call failed: %s", err_msg)
            # Differentiate timeout vs general API error
            if "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
                raise GenerationTimeoutError(f"Gemini generation timed out: {err_msg}") from e
            raise GenerationError(f"Gemini generation error: {err_msg}") from e


gemini_client = GeminiClient()
