from unittest.mock import patch, MagicMock
from app import Me

@patch('app.get_llm')
@patch('app.RAGManager')
def test_format_context(mock_rag, mock_get_llm):
    """Test that context chunks are formatted with header and proper separation."""
    mock_get_llm.return_value = (MagicMock(), "test-model")
    
    me = Me()
    chunks = ["Expert in Python.", "Built a RAG system."]
    formatted = me.format_context(chunks)
    
    assert "Here is relevant context from Mohamed Abdelkrim SENOUCI's portfolio documents:" in formatted
    assert "Expert in Python." in formatted
    assert "Built a RAG system." in formatted

@patch('app.get_llm')
@patch('app.RAGManager')
def test_format_context_empty(mock_rag, mock_get_llm):
    """Test that empty context returns empty string."""
    mock_get_llm.return_value = (MagicMock(), "test-model")
    
    me = Me()
    assert me.format_context([]) == ""

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_system_prompt_no_context(mock_rag, mock_get_llm):
    """Test standard system prompt without context."""
    mock_get_llm.return_value = (MagicMock(), "test-model")
    
    me = Me()
    prompt = me.system_prompt()
    assert "You are acting as Mohamed Abdelkrim SENOUCI" in prompt
    assert "### CONTEXT ###" not in prompt

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_system_prompt_with_context(mock_rag, mock_get_llm):
    """Test system prompt with injected context."""
    mock_get_llm.return_value = (MagicMock(), "test-model")
    
    me = Me()
    context = "Experienced in Python and AI projects."
    prompt = me.system_prompt(context=context)
    assert context in prompt
    assert "### CONTEXT ###" in prompt
    assert "Mohamed Abdelkrim SENOUCI" in prompt
