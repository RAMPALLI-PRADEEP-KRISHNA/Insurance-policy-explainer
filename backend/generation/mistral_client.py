import time
from typing import Optional
from mistralai.client import Mistral
from mistralai.client.models import UserMessage, SystemMessage
from backend.generation.llm_provider import LLMProvider
from backend.config import AppConfig
from backend.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    MistralRateLimitError,
    MistralAuthError,
    MistralTierError,
    MistralServiceUnavailableError,
)


class MistralProvider(LLMProvider):
    def __init__(self, app_config: AppConfig):
        self.config = app_config
        timeout_s = getattr(self.config, "LLM_REQUEST_TIMEOUT_SECONDS", 15)
        self.timeout_ms = int(timeout_s * 1000)
        self.client = Mistral(api_key=self.config.MISTRAL_API_KEY, timeout_ms=self.timeout_ms)
        self.model = getattr(self.config, "MISTRAL_MODEL", "ministral-8b-latest")

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        temperature: Optional[float] = None,
        response_schema: Optional[type] = None,
    ) -> str:
        messages = [
            SystemMessage(content=system_instruction),
            UserMessage(content=prompt),
        ]

        start_time = time.time()
        _RETRY_MAX_ATTEMPTS = 2
        _RETRY_BASE_DELAY_S = 1.0

        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            elapsed = time.time() - start_time
            if elapsed >= self.config.LLM_TOTAL_DEADLINE_SECONDS:
                print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: deadline, Attempts: {attempt - 1}")
                raise GenerationTimeoutError(f"Mistral generation timed out after {elapsed:.1f}s")

            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                )
                print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: success, Attempts: {attempt}")
                return response.choices[0].message.content

            except Exception as e:
                err_msg = str(e).lower()

                # Model tier restriction (HTTP 403 tier_not_allowed)
                if "tier_not_allowed" in err_msg or "not available in your subscription tier" in err_msg:
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: tier_not_allowed, Attempts: {attempt}")
                    raise MistralTierError(f"Mistral model '{self.model}' is not available in your subscription tier: {err_msg}") from e

                # Permanent authentication / authorization error (401 / bad key)
                if "401" in err_msg or "unauthorized" in err_msg or "unauthenticated" in err_msg or ("403" in err_msg and "forbidden" in err_msg):
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: auth_error, Attempts: {attempt}")
                    raise MistralAuthError(f"Mistral authentication error: {err_msg}") from e

                # Rate limit (429) — non-retryable, fail fast so fallback provider has remaining budget
                is_429 = "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg
                if is_429:
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: rate_limited, Attempts: {attempt}")
                    raise MistralRateLimitError(f"Mistral rate limit exceeded: {err_msg}") from e

                # Timeout — non-retryable, fail fast
                is_timeout = "timeout" in err_msg or "deadline" in err_msg or isinstance(e, TimeoutError)
                if is_timeout:
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: timeout, Attempts: {attempt}")
                    raise GenerationTimeoutError(f"Mistral generation timed out: {err_msg}") from e

                # Transient 503 / 502 service unavailable
                is_503 = "503" in err_msg or "unavailable" in err_msg or "502" in err_msg
                if not is_503:
                    raise GenerationError(f"Mistral generation error: {err_msg}") from e

                # Retry logic for transient 503 only if remaining budget allows
                remaining_budget = self.config.LLM_TOTAL_DEADLINE_SECONDS - (time.time() - start_time)
                if attempt < _RETRY_MAX_ATTEMPTS and remaining_budget > 3.0:
                    delay = min(_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), 1.0)
                    time.sleep(delay)
                    continue
                else:
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: unavailable, Attempts: {attempt}")
                    raise MistralServiceUnavailableError(f"Mistral service unavailable: {err_msg}") from e

        raise GenerationError("Mistral generation failed")
