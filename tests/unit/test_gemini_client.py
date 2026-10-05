"""Unit tests for Gemini client wrapper and error handling."""

from unittest.mock import MagicMock
import pytest
from src.generation.gemini_client import GeminiClient
from src.utils.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    MalformedOutputError,
    MissingAPIKeyError,
    ModelUnavailableError,
)


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
