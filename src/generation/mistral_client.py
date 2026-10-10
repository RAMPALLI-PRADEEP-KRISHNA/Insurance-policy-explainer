import time
from typing import Optional
from mistralai.client import Mistral
from mistralai.client.models import UserMessage, SystemMessage
from src.generation.llm_provider import LLMProvider
from src.config import AppConfig
from src.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    MistralRateLimitError,
    MistralAuthError,
    MistralServiceUnavailableError
)

class MistralProvider(LLMProvider):
    def __init__(self, app_config: AppConfig):
        self.config = app_config
        self.client = Mistral(api_key=self.config.MISTRAL_API_KEY)
        self.model = getattr(self.config, "MISTRAL_MODEL", "mistral-small-latest")

    def generate(self, prompt: str, system_instruction: str, temperature: Optional[float] = None, response_schema: Optional[type] = None) -> str:
        messages = [
            SystemMessage(content=system_instruction),
            UserMessage(content=prompt)
        ]
        
        start_time = time.time()
        
        # Mistral uses max 2 attempts as well
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

                # Auth error
                if "401" in err_msg or "403" in err_msg or "unauthorized" in err_msg:
                    raise MistralAuthError(f"Mistral authentication error: {err_msg}") from e

                is_timeout = "timeout" in err_msg or "deadline" in err_msg
                is_503 = "503" in err_msg or "unavailable" in err_msg or "502" in err_msg
                is_429 = "429" in err_msg or "rate limit" in err_msg

                if is_429:
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: rate_limited, Attempts: {attempt}")
                    raise MistralRateLimitError(f"Mistral rate limit exceeded: {err_msg}") from e

                if not (is_timeout or is_503):
                    raise GenerationError(f"Mistral generation error: {err_msg}") from e

                # Retry logic
                if attempt < _RETRY_MAX_ATTEMPTS:
                    delay = min(_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), 1.0)
                    time.sleep(delay)
                    continue
                else:
                    outcome = "timeout" if is_timeout else "unavailable"
                    print(f"[TIMING] LLM request - Provider: Mistral, Model: {self.model}, Outcome: {outcome}, Attempts: {attempt}")
                    if is_timeout:
                        raise GenerationTimeoutError(f"Mistral generation timed out: {err_msg}") from e
                    else:
                        raise MistralServiceUnavailableError(f"Mistral service unavailable: {err_msg}") from e

        raise GenerationError("Mistral generation failed")
