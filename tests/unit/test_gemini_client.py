import time
from unittest.mock import MagicMock, patch, call
import pytest

from backend.generation.gemini_client import GeminiClient, _RETRY_MAX_ATTEMPTS, _RETRY_BASE_DELAY_S
from backend.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    GeminiServiceUnavailableError,
    MalformedOutputError,
    MissingAPIKeyError,
    ModelUnavailableError,
)

def _make_client(side_effects):
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = side_effects
    return GeminiClient(api_key="fake-key", client_instance=mock_sdk)

def _503_exception():
    return Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'high demand', 'status': 'UNAVAILABLE'}}")

def test_gemini_client_missing_api_key_raises():
    client = GeminiClient(api_key=None, app_config=MagicMock(GEMINI_API_KEY=None, GEMINI_MODEL="gemini-2.5-flash"))
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY is not configured"):
        _ = client.client

def test_gemini_client_model_validation_failure_raises():
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.get.side_effect = Exception("Model not found: 404")
    client = GeminiClient(api_key="fake-api-key", model_name="non-existent-model", client_instance=mock_sdk_client)
    with pytest.raises(ModelUnavailableError, match="unavailable or invalid"):
        client.validate_model()

def test_gemini_client_generation_timeout_raises():
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = TimeoutError("Deadline exceeded")
    client = GeminiClient(api_key="fake-api-key", client_instance=mock_sdk_client)
    with pytest.raises(GenerationTimeoutError):
        client.generate(prompt="What is the room rent limit?")

def test_gemini_client_malformed_output_raises():
    mock_sdk_client = MagicMock()
    mock_response = MagicMock(text="")
    mock_sdk_client.models.generate_content.return_value = mock_response
    client = GeminiClient(api_key="fake-api-key", client_instance=mock_sdk_client)
    with pytest.raises(MalformedOutputError):
        client.generate(prompt="Any policy question?")

def test_gemini_client_successful_generation():
    mock_sdk_client = MagicMock()
    mock_response = MagicMock(text="The waiting period is 36 months of continuous coverage.")
    mock_sdk_client.models.generate_content.return_value = mock_response
    client = GeminiClient(api_key="fake-api-key", client_instance=mock_sdk_client)
    result = client.generate(prompt="What is the waiting period?")
    assert result == "The waiting period is 36 months of continuous coverage."

@patch("backend.generation.gemini_client.time.sleep", return_value=None)
def test_retry_succeeds_after_first_503(mock_sleep):
    mock_sdk = MagicMock()
    ok_response = MagicMock(text="The waiting period is 36 months.")
    mock_sdk.models.generate_content.side_effect = [_503_exception(), ok_response]
    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)
    result = client.generate(prompt="Question?")
    assert result == "The waiting period is 36 months."
    assert mock_sdk.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()

@patch("backend.generation.gemini_client.time.sleep", return_value=None)
def test_all_attempts_503_raises_service_unavailable(mock_sleep):
    client = _make_client([_503_exception(), _503_exception()])
    with pytest.raises(GeminiServiceUnavailableError) as exc_info:
        client.generate(prompt="What is the room rent limit?")
    error_msg = str(exc_info.value)
    assert "Gemini service unavailable" in error_msg
    assert mock_sleep.call_count == 1

@patch("backend.generation.gemini_client.time.sleep", return_value=None)
def test_exponential_backoff_delays(mock_sleep):
    client = _make_client([_503_exception(), _503_exception()])
    with pytest.raises(GeminiServiceUnavailableError):
        client.generate(prompt="Test backoff?")
    calls = mock_sleep.call_args_list
    assert len(calls) == 1
    assert calls[0] == call(1.0)

@patch("backend.generation.gemini_client.time.sleep", return_value=None)
def test_auth_error_no_retry(mock_sleep):
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = Exception("401 UNAUTHENTICATED. Invalid API key.")
    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)
    with pytest.raises(GenerationError, match="authentication error"):
        client.generate(prompt="Hi")
    assert mock_sdk.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()

@patch("backend.generation.gemini_client.time.sleep", return_value=None)
def test_other_api_error_no_retry(mock_sleep):
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = Exception("400 INVALID_ARGUMENT. Request body malformed.")
    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)
    with pytest.raises(GenerationError):
        client.generate(prompt="Hi")
    assert mock_sdk.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()

def test_model_lookup_failure_raises_model_unavailable():
    mock_sdk = MagicMock()
    mock_sdk.models.get.side_effect = Exception("404 Model not found")
    client = GeminiClient(api_key="fake-key", model_name="made-up-model", client_instance=mock_sdk)
    with pytest.raises(ModelUnavailableError, match="unavailable or invalid"):
        client.validate_model()


def test_is_transient_503_classifier():
    client = GeminiClient(api_key='fake')
    assert client._is_transient_503('503 unavailable') is True
    assert client._is_transient_503('503 service unavailable') is True
    assert client._is_transient_503('500 internal error') is False

def test_is_auth_error_classifier():
    client = GeminiClient(api_key='fake')
    assert client._is_auth_error('401 unauthenticated') is True
    assert client._is_auth_error('403 permission denied') is True
    assert client._is_auth_error('invalid api_key') is True
    assert client._is_auth_error('404 not found') is False


