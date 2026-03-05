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
        from openai import OpenAI
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    elif backend == "openai":
        from openai import OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif backend == "gemini":
        from openai import OpenAI
        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        client = OpenAI(
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
    elif backend == "grok":
        from openai import OpenAI
        model = os.getenv("GROK_MODEL", "grok-4.1-fast")
        client = OpenAI(
            base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            api_key=os.getenv("GROK_API_KEY")
        )
    elif backend == "groq":
        from openai import OpenAI
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = OpenAI(
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif backend == "openrouter":
        from openai import OpenAI
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    elif backend == "mistral":
        from openai import OpenAI
        model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        client = OpenAI(
            base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            api_key=os.getenv("MISTRAL_API_KEY")
        )
    elif backend == "ollama":
        from openai import OpenAI
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            # Ollama doesn't validate keys, but the OpenAI client expects a string.
            api_key=os.getenv("OLLAMA_API_KEY", "ollama")
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return client, model
