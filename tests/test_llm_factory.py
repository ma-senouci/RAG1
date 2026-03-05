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

def test_get_llm_gemini_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Gemini."""
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "gemini-3.1-flash-lite-preview"
    assert client.api_key == "test-key-gemini"
    assert "generativelanguage.googleapis.com" in str(client.base_url)

def test_get_llm_grok_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Grok."""
    monkeypatch.setenv("LLM_BACKEND", "grok")
    monkeypatch.setenv("GROK_API_KEY", "test-key-grok")
    monkeypatch.delenv("GROK_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "grok-4.1-fast"
    assert client.api_key == "test-key-grok"
    assert str(client.base_url) == "https://api.x.ai/v1/"

def test_get_llm_groq_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Groq."""
    monkeypatch.setenv("LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-groq")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "llama-3.3-70b-versatile"
    assert client.api_key == "test-key-groq"
    assert "api.groq.com" in str(client.base_url)

def test_get_llm_openrouter_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for OpenRouter."""
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-or")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "anthropic/claude-haiku-4.5"
    assert client.api_key == "test-key-or"
    assert "openrouter.ai" in str(client.base_url)

def test_get_llm_mistral_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Mistral."""
    monkeypatch.setenv("LLM_BACKEND", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-mistral")
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "mistral-small-latest"
    assert client.api_key == "test-key-mistral"
    assert str(client.base_url) == "https://api.mistral.ai/v1/"

def test_get_llm_ollama_default(monkeypatch):
    """Verify factory returns correct hardcoded defaults for Ollama."""
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    
    client, model = get_llm()
    assert model == "llama3.1:8b-instruct-q4_0"
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
