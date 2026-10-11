import time
from typing import Optional, List
from backend.generation.llm_provider import LLMProvider
from backend.config import AppConfig
from backend.utils.logging_utils import get_logger
from backend.utils.exceptions import (
    GenerationTimeoutError,
    MistralRateLimitError,
    GeminiRateLimitError,
    MistralServiceUnavailableError,
    GeminiServiceUnavailableError,
    MistralAuthError,
    MistralTierError,
    ModelUnavailableError,
    MissingAPIKeyError,
    GenerationError,
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
            from backend.generation.mistral_client import MistralProvider
            return MistralProvider(app_config=self.config)
        elif name == "gemini":
            if not getattr(self.config, "GEMINI_API_KEY", None):
                return None
            from backend.generation.gemini_client import GeminiClient
            return GeminiClient(app_config=self.config)
        return None

    @property
    def model(self):
        return self.last_used_model or "provider-chain"

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        temperature: Optional[float] = None,
        response_schema: Optional[type] = None,
    ) -> str:
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
                logger.warning(
                    "[LLM] Total deadline (%ss) reached before calling %s",
                    self.config.LLM_TOTAL_DEADLINE_SECONDS,
                    provider_name,
                )
                break

            try:
                response = client.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_schema=response_schema,
                )

                # Report which provider and model was used
                model_name = getattr(client, "model", getattr(client, "model_name", "unknown"))
                self.last_used_model = f"{provider_name}/{model_name}"
                return response

            except (
                MistralRateLimitError,
                GeminiRateLimitError,
                MistralServiceUnavailableError,
                GeminiServiceUnavailableError,
                GenerationTimeoutError,
                MistralAuthError,
                MistralTierError,
                ModelUnavailableError,
                MissingAPIKeyError,
                GenerationError,
            ) as e:
                last_error = e

                # Determine concise, sanitized reason
                if isinstance(e, MistralTierError) or "tier_not_allowed" in str(e).lower():
                    reason = "tier_not_allowed"
                elif isinstance(e, (MistralRateLimitError, GeminiRateLimitError)) or "rate" in str(e).lower():
                    reason = "rate_limited"
                elif isinstance(e, (MistralServiceUnavailableError, GeminiServiceUnavailableError)) or "unavailable" in str(e).lower():
                    reason = "service_unavailable"
                elif isinstance(e, GenerationTimeoutError) or "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    reason = "timeout"
                elif isinstance(e, MistralAuthError) or "auth" in str(e).lower():
                    reason = "auth_error"
                elif isinstance(e, ModelUnavailableError):
                    reason = "model_unavailable"
                elif isinstance(e, MissingAPIKeyError):
                    reason = "missing_key"
                else:
                    reason = "failed"

                error_reasons.append(f"{provider_name}: {reason}")

                if len(providers) > 1 and provider_name == providers[0][0]:
                    logger.warning("[LLM] %s failed (%s), trying %s", provider_name, reason, providers[1][0])

        # If we exited the loop, all attempted providers failed or total deadline was hit
        if last_error:
            combined_reason = "; ".join(error_reasons)
            if combined_reason:
                last_error.args = (f"{last_error.args[0]} ({combined_reason})",) + last_error.args[1:]
            raise last_error

        elapsed = time.time() - start_time
        raise GenerationTimeoutError(f"Total deadline exceeded after {elapsed:.1f}s")
