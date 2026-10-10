import pytest
import time
from unittest.mock import patch, MagicMock

from src.generation.provider_chain import ProviderChain
from src.generation.answer_generator import get_llm_provider, AnswerGenerator
from src.config import AppConfig
from src.utils.exceptions import (
    MistralRateLimitError,
    GeminiRateLimitError,
    MistralServiceUnavailableError,
    MistralAuthError,
    GenerationTimeoutError,
    MissingAPIKeyError,
    GenerationError
)
from src.models.data_models import Evidence, EvidenceStatus

@pytest.fixture
def mock_config():
    config = AppConfig()
    config.LLM_PROVIDER = "mistral"
    config.LLM_FALLBACK_PROVIDER = "gemini"
    config.MISTRAL_API_KEY = "fake-mistral"
    config.GEMINI_API_KEY = "fake-gemini"
    config.LLM_TOTAL_DEADLINE_SECONDS = 40
    return config

def test_mistral_succeeds_gemini_not_called(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.return_value = "Mistral Answer"
        mock_m.model = "mistral-small-latest"
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        ans = chain.generate("prompt", "sys")
        
        assert ans == "Mistral Answer"
        assert chain.last_used_model == "mistral/mistral-small-latest"
        mock_m.generate.assert_called_once()
        mock_g.generate.assert_not_called()

def test_mistral_429_gemini_answers(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.side_effect = MistralRateLimitError("429")
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_g.generate.return_value = "Gemini Answer"
        mock_g.model = "gemini-3.8-flash"
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        ans = chain.generate("p", "s")
        
        assert ans == "Gemini Answer"
        assert chain.last_used_model == "gemini/gemini-3.8-flash"
        mock_g.generate.assert_called_once()

def test_mistral_503_gemini_answers(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.side_effect = MistralServiceUnavailableError("503")
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_g.generate.return_value = "Gemini Answer"
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        ans = chain.generate("p", "s")
        assert ans == "Gemini Answer"

def test_mistral_timeout_gemini_answers(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.side_effect = GenerationTimeoutError("timeout")
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_g.generate.return_value = "Gemini Answer"
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        ans = chain.generate("p", "s")
        assert ans == "Gemini Answer"

def test_mistral_auth_error_gemini_answers(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.side_effect = MistralAuthError("401")
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_g.generate.return_value = "Gemini Answer"
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        ans = chain.generate("p", "s")
        assert ans == "Gemini Answer"

def test_both_fail_llm_unavailable(mock_config):
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        mock_m = MagicMock()
        mock_m.generate.side_effect = MistralRateLimitError("mistral rate 429")
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_g.generate.side_effect = GeminiRateLimitError("gemini rate 429")
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        generator = AnswerGenerator(llm_provider_instance=chain)
        
        evidence = Evidence(doc_id="d1", query="q", status=EvidenceStatus.SUFFICIENT, chunks=[])
        ans = generator.generate_answer("q", evidence)
        
        assert getattr(ans, "llm_unavailable", False) is True
        reason = getattr(ans, "llm_unavailable_reason", "")
        assert "mistral: rate_limited" in reason
        assert "gemini: rate_limited" in reason

def test_missing_mistral_key_gemini_only(mock_config):
    mock_config.MISTRAL_API_KEY = None
    with patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        mock_g = MagicMock()
        mock_g.generate.return_value = "Gemini Answer"
        mock_g.model = "gemini"
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        assert chain.primary_client is None
        ans = chain.generate("p", "s")
        
        assert ans == "Gemini Answer"
        assert chain.last_used_model == "gemini/gemini"

def test_total_deadline_respected(mock_config, monkeypatch):
    mock_config.LLM_TOTAL_DEADLINE_SECONDS = 0.5
    
    with patch("src.generation.mistral_client.MistralProvider") as mock_mistral, \
         patch("src.generation.gemini_client.GeminiClient") as mock_gemini:
        
        def slow_generate(*args, **kwargs):
            time.sleep(0.6)
            raise MistralRateLimitError("429")

        mock_m = MagicMock()
        mock_m.generate.side_effect = slow_generate
        mock_mistral.return_value = mock_m
        
        mock_g = MagicMock()
        mock_gemini.return_value = mock_g

        chain = ProviderChain(mock_config)
        
        with pytest.raises(MistralRateLimitError):
            chain.generate("p", "s")
            
        # Gemini should not be called because time is up
        mock_g.generate.assert_not_called()

def test_provider_fake_never_builds_chain(mock_config):
    mock_config.LLM_PROVIDER = "fake"
    provider = get_llm_provider(mock_config)
    
    assert not isinstance(provider, ProviderChain)
    assert type(provider).__name__ == "GeminiClient"
