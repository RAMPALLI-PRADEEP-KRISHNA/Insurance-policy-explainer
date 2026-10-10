import time
from typing import Optional, List
from src.generation.llm_provider import LLMProvider
from src.config import AppConfig
from src.utils.logging_utils import get_logger
from src.utils.exceptions import (
    GenerationTimeoutError,
    MistralRateLimitError,
    GeminiRateLimitError,
    MistralServiceUnavailableError,
    GeminiServiceUnavailableError,
    MistralAuthError,
    MissingAPIKeyError,
    GenerationError
)

logger = get_logger(__name__)

class ProviderChain(LLMProvider):
    def __init__(self, config: AppConfig):
        self.config = config
        self.primary_name = config.LLM_PROVIDER
        self.fallback_name = getattr(config, "LLM_FALLBACK_PROVIDER", "gemini")
        
        self.primary_client = self._instantiate_client(self.primary_name)
        self.fallback_client = self._instantiate_client(self.fallback_name)
        self.last_used_model = None

    def _instantiate_client(self, name: str) -> Optional[LLMProvider]:
        if name == "mistral":
            if not self.config.MISTRAL_API_KEY:
                return None
            from src.generation.mistral_client import MistralProvider
            return MistralProvider(app_config=self.config)
        elif name == "gemini":
            if not getattr(self.config, "GEMINI_API_KEY", None):
                return None
            from src.generation.gemini_client import GeminiClient
            return GeminiClient(app_config=self.config)
        return None

    @property
    def model(self):
        return self.last_used_model or "provider-chain"

    def generate(self, prompt: str, system_instruction: str, temperature: Optional[float] = None, response_schema: Optional[type] = None) -> str:
        start_time = time.time()
        providers = []
        if self.primary_client:
            providers.append((self.primary_name, self.primary_client))
        if self.fallback_client and self.fallback_client is not self.primary_client:
            providers.append((self.fallback_name, self.fallback_client))

        if not providers:
            raise MissingAPIKeyError("No LLM provider keys configured.")

        last_error = None
        error_reasons = []

        for provider_name, client in providers:
            elapsed = time.time() - start_time
            if elapsed >= self.config.LLM_TOTAL_DEADLINE_SECONDS:
                break
            
            try:
                response = client.generate(
                    prompt=prompt, 
                    system_instruction=system_instruction, 
                    temperature=temperature, 
                    response_schema=response_schema
                )
                
                # Report which provider and model was used
                model_name = getattr(client, "model", getattr(client, "model_name", "unknown"))
                self.last_used_model = f"{provider_name}/{model_name}"
                return response

            except (
                MistralRateLimitError, GeminiRateLimitError, 
                MistralServiceUnavailableError, GeminiServiceUnavailableError, 
                GenerationTimeoutError, MistralAuthError, MissingAPIKeyError
            ) as e:
                # Fall through to next provider
                last_error = e
                err_str = str(e).lower()
                reason = "rate_limited" if "rate" in err_str else ("service_unavailable" if "unavailable" in err_str else ("timeout" if "time" in err_str else "auth_error"))
                error_reasons.append(f"{provider_name}: {reason}")
                
                if len(providers) > 1 and provider_name == providers[0][0]:
                    logger.warning(f"[LLM] {provider_name} failed ({reason}), trying {providers[1][0]}")

        # If we exited the loop, it means all providers failed or we hit the total deadline
        if last_error:
            # Append combined reasons to the exception message so answer_generator can pull it
            combined_reason = "; ".join(error_reasons)
            if combined_reason:
                last_error.args = (f"{last_error.args[0]} ({combined_reason})",) + last_error.args[1:]
            raise last_error

        elapsed = time.time() - start_time
        raise GenerationTimeoutError(f"Total deadline exceeded after {elapsed:.1f}s")
