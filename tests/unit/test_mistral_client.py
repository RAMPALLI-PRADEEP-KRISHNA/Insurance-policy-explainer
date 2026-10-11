import pytest
from unittest.mock import MagicMock, patch

from backend.generation.mistral_client import MistralProvider
from backend.config import AppConfig
from backend.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    MistralRateLimitError,
    MistralAuthError,
    MistralTierError,
    MistralServiceUnavailableError,
)


@pytest.fixture
def mock_config():
    config = AppConfig()
    config.MISTRAL_API_KEY = "test-key"
    config.MISTRAL_MODEL = "ministral-8b-latest"
    config.LLM_REQUEST_TIMEOUT_SECONDS = 15
    config.LLM_TOTAL_DEADLINE_SECONDS = 30
    return config


def test_mistral_client_successful_generation(mock_config):
    provider = MistralProvider(mock_config)
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Sample Mistral Response"))]
    provider.client.chat.complete = MagicMock(return_value=mock_resp)

    res = provider.generate("What is the deductible?", "system instruction")
    assert res == "Sample Mistral Response"
    provider.client.chat.complete.assert_called_once()


def test_mistral_tier_error_fails_immediately_without_retry(mock_config):
    provider = MistralProvider(mock_config)
    provider.client.chat.complete = MagicMock(
        side_effect=Exception('Status 403. Body: {"type":"tier_not_allowed","message":"This model is not available in your subscription tier"}')
    )

    with pytest.raises(MistralTierError) as exc_info:
        provider.generate("prompt", "system")

    assert "subscription tier" in str(exc_info.value)
    assert provider.client.chat.complete.call_count == 1


def test_mistral_auth_error_fails_immediately_without_retry(mock_config):
    provider = MistralProvider(mock_config)
    provider.client.chat.complete = MagicMock(
        side_effect=Exception("Status 401: Unauthorized API key")
    )

    with pytest.raises(MistralAuthError):
        provider.generate("prompt", "system")

    assert provider.client.chat.complete.call_count == 1


def test_mistral_rate_limit_fails_immediately_without_retry(mock_config):
    provider = MistralProvider(mock_config)
    provider.client.chat.complete = MagicMock(
        side_effect=Exception("Status 429: Rate limit exceeded")
    )

    with pytest.raises(MistralRateLimitError):
        provider.generate("prompt", "system")

    assert provider.client.chat.complete.call_count == 1


def test_mistral_timeout_fails_immediately_without_retry(mock_config):
    provider = MistralProvider(mock_config)
    provider.client.chat.complete = MagicMock(
        side_effect=TimeoutError("Request timed out after 15s")
    )

    with pytest.raises(GenerationTimeoutError):
        provider.generate("prompt", "system")

    assert provider.client.chat.complete.call_count == 1


@patch("backend.generation.mistral_client.time.sleep", return_value=None)
def test_mistral_transient_503_retries_and_succeeds(mock_sleep, mock_config):
    provider = MistralProvider(mock_config)
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Success on retry"))]
    provider.client.chat.complete = MagicMock(
        side_effect=[Exception("503 Service Unavailable"), mock_resp]
    )

    res = provider.generate("prompt", "system")
    assert res == "Success on retry"
    assert provider.client.chat.complete.call_count == 2
    mock_sleep.assert_called_once()


@patch("backend.generation.mistral_client.time.sleep", return_value=None)
def test_mistral_transient_503_exhausted_raises_service_unavailable(mock_sleep, mock_config):
    provider = MistralProvider(mock_config)
    provider.client.chat.complete = MagicMock(
        side_effect=[Exception("503 Service Unavailable"), Exception("503 Service Unavailable")]
    )

    with pytest.raises(MistralServiceUnavailableError):
        provider.generate("prompt", "system")

    assert provider.client.chat.complete.call_count == 2
    assert mock_sleep.call_count == 1
