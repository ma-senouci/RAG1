import pytest
from unittest.mock import patch, MagicMock
from app import Me

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_system_prompt_no_context(mock_rag, mock_openai):
    """Test standard system prompt without context."""
    me = Me()
    prompt = me.system_prompt()
    assert "You are acting as Mohamed Abdelkrim SENOUCI" in prompt
    assert "## Contextual Evidence" not in prompt

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_system_prompt_with_context(mock_rag, mock_openai):
    """Test system prompt with injected context."""
    me = Me()
    context = "## Contextual Evidence (from professional documents):\n\nTest Context"
    prompt = me.system_prompt(context=context)
    assert context in prompt
    assert "Mohamed Abdelkrim SENOUCI" in prompt
