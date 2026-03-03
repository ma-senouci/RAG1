import os

def get_llm(backend=None):
    """
    Returns an LLM client and model name based on the specified backend.
    
    Logic:
    1. Use provider-specific environment variable (e.g., OPENAI_MODEL).
    2. Fallback to hardcoded default (e.g., 'gpt-4.1-mini').
    """
    if backend is None:
        backend = os.getenv("LLM_BACKEND", "deepseek").lower()

    if backend == "deepseek":
        from openai import OpenAI as DeepSeekClient
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = DeepSeekClient(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    elif backend == "openai":
        from openai import OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif backend == "ollama":
        from openai import OpenAI  # OpenAI-compatible API
        model = os.getenv("OLLAMA_MODEL", "llama3")
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            # Ollama doesn't validate keys, but the OpenAI client expects a string.
            api_key=os.getenv("OLLAMA_API_KEY", "ollama")
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return client, model
