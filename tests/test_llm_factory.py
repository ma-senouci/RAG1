import pytest
from llm_factory import get_llm

def test_get_llm_deepseek_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for DeepSeek."""
    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "deepseek-chat"
    assert client.api_key == "test-key"
    assert str(client.base_url) == "https://api.deepseek.com/v1/"

def test_get_llm_openai_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for OpenAI."""
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-oa")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "gpt-4.1-mini"
    assert client.api_key == "test-key-oa"

def test_get_llm_ollama_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Ollama."""
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "llama3"
    assert client.api_key == "ollama"

def test_get_llm_provider_specific_override(monkeypatch):
    """Verify provider-specific variables (e.g., OPENAI_MODEL) work."""
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "target-model")
    
    _, model = get_llm()
    assert model == "target-model"

def test_get_llm_invalid():
    """Verify factory raises ValueError for unknown backends."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_llm(backend="unknown")
