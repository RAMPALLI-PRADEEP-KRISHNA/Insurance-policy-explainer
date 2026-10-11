"""Gemini API client wrapper using the modern Google GenAI SDK (google-genai)."""

import os
import time
from typing import Optional, Any
from backend.config import AppConfig, config as default_config
from backend.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    GeminiServiceUnavailableError,
    GeminiRateLimitError,
    MalformedOutputError,
    MissingAPIKeyError,
    ModelUnavailableError,
)
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Transient 503 retry policy — only for service-overload responses
_RETRY_MAX_ATTEMPTS = 2   # 1 initial + 1 retry
_RETRY_BASE_DELAY_S = 2.0


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
        """Lazy-initialize the genai Client with configured request timeout."""
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
                )
            try:
                from google import genai
                from google.genai import types

                timeout_s = getattr(self.config, "LLM_REQUEST_TIMEOUT_SECONDS", 15)
                timeout_ms = int(timeout_s * 1000)
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
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

    @staticmethod
    def _is_transient_503(err_str: str) -> bool:
        """Return True if the error string indicates a transient 503 server overload."""
        return "503" in err_str and "unavailable" in err_str.lower()

    @staticmethod
    def _is_auth_error(err_str: str) -> bool:
        """Return True for permanent authentication/permission failures (401, 403)."""
        return "401" in err_str or "403" in err_str or "api_key" in err_str.lower()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[type] = None,
    ) -> str:
        """
        Generate content using the configured Gemini model.

        Retries at most once on transient 503 UNAVAILABLE errors if time budget allows.
        Non-transient errors (401/403, 429 quota, timeouts) are never retried.

        Args:
            prompt: Grounded user prompt including evidence and question.
            system_instruction: Strict grounding instructions.
            temperature: Sampling temperature (default: config.GEMINI_TEMPERATURE).
            response_schema: Optional Pydantic model for structured JSON output.

        Returns:
            Generated text content string.

        Raises:
            MissingAPIKeyError: If API key is missing.
            GenerationTimeoutError: If the call times out.
            MalformedOutputError: If the model returns an empty or invalid response.
            GeminiServiceUnavailableError: If the Gemini API is persistently unavailable (503).
            GenerationError: On general API or network failures.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        temp = temperature if temperature is not None else self.config.GEMINI_TEMPERATURE
        start_time = time.time()
        last_err_msg = ""

        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            elapsed = time.time() - start_time
            if elapsed >= self.config.LLM_TOTAL_DEADLINE_SECONDS:
                print(f"[TIMING] LLM request - Provider: Gemini, Model: {self.model_name}, Outcome: deadline, Attempts: {attempt - 1}")
                raise GenerationTimeoutError(f"Gemini generation timed out after {elapsed:.1f}s")

            try:
                from google.genai import types

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

                print(f"[TIMING] LLM request - Provider: Gemini, Model: {self.model_name}, Outcome: success, Attempts: {attempt}")
                return response.text.strip()

            except (MissingAPIKeyError, MalformedOutputError, GenerationTimeoutError):
                raise
            except Exception as e:
                err_msg = str(e).lower()
                last_err_msg = err_msg

                # Permanent authentication failure (401 / 403 / bad key) — do not retry
                if self._is_auth_error(err_msg):
                    logger.error("Gemini authentication failure: %s", err_msg)
                    raise GenerationError(f"Gemini authentication error: {err_msg}") from e

                # Rate limit / quota exhausted (429) — do not retry
                is_429 = "429" in err_msg or "resource exhausted" in err_msg or "quota" in err_msg
                if is_429:
                    print(f"[TIMING] LLM request - Provider: Gemini, Model: {self.model_name}, Outcome: rate_limited, Attempts: {attempt}")
                    raise GeminiRateLimitError(f"Gemini rate limit exceeded: {err_msg}") from e

                # Timeouts (504, TimeoutError, deadline exceeded) — non-retryable, fail fast
                is_timeout = (
                    isinstance(e, TimeoutError)
                    or "timeout" in err_msg
                    or "deadline" in err_msg
                    or "504" in err_msg
                )
                if is_timeout:
                    print(f"[TIMING] LLM request - Provider: Gemini, Model: {self.model_name}, Outcome: timeout, Attempts: {attempt}")
                    raise GenerationTimeoutError(f"Gemini generation timed out: {err_msg}") from e

                # Transient 503 check
                is_503 = self._is_transient_503(err_msg)
                if not is_503:
                    logger.error("Gemini API call failed unrecoverably: %s", err_msg)
                    raise GenerationError(f"Gemini generation error: {err_msg}") from e

                # Transient 503 retry logic — only if budget permits
                remaining_budget = self.config.LLM_TOTAL_DEADLINE_SECONDS - (time.time() - start_time)
                if attempt < _RETRY_MAX_ATTEMPTS and remaining_budget > 3.0:
                    delay = min(_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), 1.0)
                    time.sleep(delay)
                    continue
                else:
                    outcome = "unavailable"
                    print(f"[TIMING] LLM request - Provider: Gemini, Model: {self.model_name}, Outcome: {outcome}, Attempts: {attempt}")
                    raise GeminiServiceUnavailableError(f"Gemini service unavailable: {err_msg}") from e

        raise GenerationError(f"Gemini generation failed: {last_err_msg}")


gemini_client = GeminiClient()
