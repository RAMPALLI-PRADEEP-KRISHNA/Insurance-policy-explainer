import time
import pytest
from unittest.mock import MagicMock, patch

from backend.config import config
from backend.generation.gemini_client import GeminiClient
from backend.utils.exceptions import GenerationTimeoutError, GeminiServiceUnavailableError, GeminiRateLimitError

def test_hangs_longer_than_timeout(monkeypatch):
    """Test that a TimeoutError triggers the fallback and finishes quickly."""
    client = GeminiClient()
    
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = TimeoutError("Simulated timeout")
    monkeypatch.setattr(client, "_client", mock_genai_client)
    
    start = time.time()
    with pytest.raises(GenerationTimeoutError):
        client.generate(prompt="Evidence Excerpts from Policy:\n[Excerpt]\n\nInstruction:")
    duration = time.time() - start
    assert duration < 35

def test_429_rate_limit(monkeypatch):
    """Test that 429 quota errors fail immediately without retry."""
    client = GeminiClient()
    
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = Exception("429 Resource Exhausted")
    monkeypatch.setattr(client, "_client", mock_genai_client)
    
    start = time.time()
    with pytest.raises(GeminiRateLimitError):
        client.generate(prompt="Evidence Excerpts from Policy:\n[Excerpt]\n\nInstruction:")
    duration = time.time() - start
    
    assert duration < 35
    assert mock_genai_client.models.generate_content.call_count == 1

def test_503_followed_by_success(monkeypatch):
    """Test 503 causes one retry, then succeeds."""
    client = GeminiClient()
    
    mock_genai_client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "Success Answer!"
    mock_genai_client.models.generate_content.side_effect = [
        Exception("503 Service Unavailable"),
        success_resp
    ]
    monkeypatch.setattr(client, "_client", mock_genai_client)
    
    start = time.time()
    result = client.generate(prompt="Evidence Excerpts from Policy:\n[Excerpt]\n\nInstruction:")
    duration = time.time() - start
    
    assert duration < 35
    assert result == "Success Answer!"
    assert mock_genai_client.models.generate_content.call_count == 2

def test_503_twice(monkeypatch):
    """Test 503 twice returns fallback."""
    client = GeminiClient()
    
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = [
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable")
    ]
    monkeypatch.setattr(client, "_client", mock_genai_client)
    
    start = time.time()
    with pytest.raises(GeminiServiceUnavailableError):
        client.generate(prompt="Evidence Excerpts from Policy:\n[Excerpt]\n\nInstruction:")
    duration = time.time() - start
    
    assert duration < 35
    assert mock_genai_client.models.generate_content.call_count == 2
