"""Unit tests for Gemini client wrapper — retry logic, error handling, and service-unavailable classification."""

import time
from unittest.mock import MagicMock, patch, call
import pytest

from src.generation.gemini_client import GeminiClient, _RETRY_MAX_ATTEMPTS, _RETRY_BASE_DELAY_S
from src.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    GeminiServiceUnavailableError,
    MalformedOutputError,
    MissingAPIKeyError,
    ModelUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(side_effects):
    """Return a GeminiClient with a mock SDK client whose generate_content raises/returns in sequence."""
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = side_effects
    return GeminiClient(api_key="fake-key", client_instance=mock_sdk)


def _503_exception():
    return Exception(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'high demand', 'status': 'UNAVAILABLE'}}"
    )


# ---------------------------------------------------------------------------
# Existing tests (kept intact, names preserved)
# ---------------------------------------------------------------------------

def test_gemini_client_missing_api_key_raises():
    """Verify MissingAPIKeyError when API key is missing and client property is accessed."""
    client = GeminiClient(api_key=None, app_config=MagicMock(GEMINI_API_KEY=None, GEMINI_MODEL="gemini-2.5-flash"))

    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY is not configured"):
        _ = client.client


def test_gemini_client_model_validation_failure_raises():
    """Verify ModelUnavailableError when model check fails."""
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.get.side_effect = Exception("Model not found: 404")

    client = GeminiClient(
        api_key="fake-api-key",
        model_name="non-existent-model",
        client_instance=mock_sdk_client,
    )

    with pytest.raises(ModelUnavailableError, match="unavailable or invalid"):
        client.validate_model()


def test_gemini_client_generation_timeout_raises():
    """Verify GenerationTimeoutError when generate_content times out."""
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = TimeoutError("Deadline exceeded")

    client = GeminiClient(
        api_key="fake-api-key",
        client_instance=mock_sdk_client,
    )

    with pytest.raises(GenerationTimeoutError, match="timed out"):
        client.generate(prompt="What is the waiting period?")


def test_gemini_client_malformed_output_raises():
    """Verify MalformedOutputError when response text is None or empty."""
    mock_sdk_client = MagicMock()
    mock_response = MagicMock(text="")
    mock_sdk_client.models.generate_content.return_value = mock_response

    client = GeminiClient(
        api_key="fake-api-key",
        client_instance=mock_sdk_client,
    )

    with pytest.raises(MalformedOutputError, match="empty or malformed"):
        client.generate(prompt="What is the waiting period?")


def test_gemini_client_successful_generation():
    """Verify successful response extraction with mock."""
    mock_sdk_client = MagicMock()
    mock_response = MagicMock(text="The waiting period is 36 months of continuous coverage.")
    mock_sdk_client.models.generate_content.return_value = mock_response

    client = GeminiClient(
        api_key="fake-api-key",
        client_instance=mock_sdk_client,
    )

    result = client.generate(prompt="What is the waiting period?")
    assert result == "The waiting period is 36 months of continuous coverage."


# ---------------------------------------------------------------------------
# New tests: retry logic and error classification
# ---------------------------------------------------------------------------

@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_retry_succeeds_after_first_503(mock_sleep):
    """First attempt returns 503; retry succeeds — generate() returns the answer."""
    mock_sdk = MagicMock()
    ok_response = MagicMock(text="The waiting period is 36 months.")
    mock_sdk.models.generate_content.side_effect = [_503_exception(), ok_response]

    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)
    result = client.generate(prompt="What is the waiting period?")

    assert result == "The waiting period is 36 months."
    assert mock_sdk.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(_RETRY_BASE_DELAY_S)  # 2.0 s delay after attempt 1


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_retry_succeeds_after_two_503s(mock_sleep):
    """First two attempts return 503; third (final) attempt succeeds."""
    mock_sdk = MagicMock()
    ok_response = MagicMock(text="Co-payment is 10% for age 61 and above.")
    mock_sdk.models.generate_content.side_effect = [
        _503_exception(), _503_exception(), ok_response
    ]

    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)
    result = client.generate(prompt="Is there a co-payment?")

    assert "10%" in result
    assert mock_sdk.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_all_attempts_503_raises_service_unavailable(mock_sleep):
    """All 3 attempts return 503 → GeminiServiceUnavailableError with clean message (no raw JSON)."""
    client = _make_client([_503_exception(), _503_exception(), _503_exception()])

    with pytest.raises(GeminiServiceUnavailableError) as exc_info:
        client.generate(prompt="What is the room rent limit?")

    error_msg = str(exc_info.value)
    assert "temporarily unavailable" in error_msg
    assert "503" not in error_msg           # raw error code NOT exposed
    assert "UNAVAILABLE" not in error_msg   # raw status NOT exposed
    assert mock_sleep.call_count == 2       # slept between 3 attempts


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_auth_error_no_retry(mock_sleep):
    """A 401/403 auth failure must not be retried — raises GenerationError immediately."""
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = Exception(
        "401 UNAUTHENTICATED. Invalid API key."
    )

    client = GeminiClient(api_key="bad-key", client_instance=mock_sdk)

    with pytest.raises(GenerationError, match="authentication error"):
        client.generate(prompt="Any question?")

    assert mock_sdk.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_other_api_error_no_retry(mock_sleep):
    """Non-transient API errors must not be retried."""
    mock_sdk = MagicMock()
    mock_sdk.models.generate_content.side_effect = Exception(
        "400 INVALID_ARGUMENT. Request body malformed."
    )

    client = GeminiClient(api_key="fake-key", client_instance=mock_sdk)

    with pytest.raises(GenerationError):
        client.generate(prompt="Some question?")

    assert mock_sdk.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


def test_model_lookup_failure_raises_model_unavailable():
    """Model lookup failure raises ModelUnavailableError (configuration error path)."""
    mock_sdk = MagicMock()
    mock_sdk.models.get.side_effect = Exception("404 Model not found")

    client = GeminiClient(api_key="fake-key", model_name="made-up-model", client_instance=mock_sdk)

    with pytest.raises(ModelUnavailableError, match="unavailable or invalid"):
        client.validate_model()


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_503_error_message_is_clean(mock_sleep):
    """GeminiServiceUnavailableError message is human-readable and safe for UI display."""
    client = _make_client([_503_exception(), _503_exception(), _503_exception()])

    with pytest.raises(GeminiServiceUnavailableError) as exc_info:
        client.generate(prompt="Any policy question?")

    msg = str(exc_info.value)
    # User-facing message must be clean
    assert "Please try again" in msg
    assert "policy was processed successfully" in msg
    # Must not contain raw API internals
    assert "UNAVAILABLE" not in msg


def test_is_transient_503_classifier():
    """_is_transient_503 correctly classifies 503 strings."""
    assert GeminiClient._is_transient_503("503 UNAVAILABLE service overloaded") is True
    assert GeminiClient._is_transient_503("503 unavailable") is True
    assert GeminiClient._is_transient_503("503 something else") is False   # no 'unavailable'
    assert GeminiClient._is_transient_503("503") is False
    assert GeminiClient._is_transient_503("401 Unauthenticated") is False


def test_is_auth_error_classifier():
    """_is_auth_error correctly classifies auth failures."""
    assert GeminiClient._is_auth_error("401 UNAUTHENTICATED") is True
    assert GeminiClient._is_auth_error("403 PERMISSION_DENIED") is True
    assert GeminiClient._is_auth_error("invalid api_key provided") is True
    assert GeminiClient._is_auth_error("503 UNAVAILABLE") is False


@patch("src.generation.gemini_client.time.sleep", return_value=None)
def test_exponential_backoff_delays(mock_sleep):
    """Retry delays use exponential backoff: 2s, 4s for attempts 1 and 2."""
    client = _make_client([_503_exception(), _503_exception(), _503_exception()])

    with pytest.raises(GeminiServiceUnavailableError):
        client.generate(prompt="Test backoff?")

    calls = mock_sleep.call_args_list
    assert len(calls) == 2
    assert calls[0] == call(_RETRY_BASE_DELAY_S * 1)   # 2.0s after attempt 1
    assert calls[1] == call(_RETRY_BASE_DELAY_S * 2)   # 4.0s after attempt 2
