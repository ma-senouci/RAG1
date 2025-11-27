import pytest
import os
from unittest.mock import MagicMock, patch
from app import Me

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_initialization_env_vars(mock_rag, mock_openai):
    """Test that Me initializes with environment variables."""
    os.environ["DEEPSEEK_API_KEY"] = "test_key"
    os.environ["DEEPSEEK_BASE_URL"] = "https://test.api.com"
    
    me = Me()
    
    mock_openai.assert_called_once_with(
        base_url="https://test.api.com",
        api_key="test_key"
    )

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_chat_orchestration_success(mock_rag_class, mock_openai_class):
    """Test standard Me.chat orchestration with RAG context."""
    # Setup mocks
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = ["context chunk"]
    mock_rag.format_context.return_value = "## Contextual Evidence: test"
    
    mock_openai = mock_openai_class.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="Hello!"))]
    mock_openai.chat.completions.create.return_value = mock_response
    
    me = Me()
    response = me.chat("Tell me about yourself", [])
    
    assert response == "Hello!"
    mock_rag.get_query_embedding.assert_called_once_with("Tell me about yourself")
    mock_rag.search.assert_called_once()
    mock_openai.chat.completions.create.assert_called_once()
    
    # Verify system prompt contains context
    args, kwargs = mock_openai.chat.completions.create.call_args
    messages = kwargs['messages']
    assert messages[0]['role'] == 'system'
    assert "## Contextual Evidence: test" in messages[0]['content']

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_chat_orchestration_fallback(mock_rag_class, mock_openai_class):
    """Test Me.chat fallback when RAG fails."""
    # Setup mocks
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.side_effect = Exception("RAG Error")
    
    mock_openai = mock_openai_class.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="Fallback Response"))]
    mock_openai.chat.completions.create.return_value = mock_response
    
    me = Me()
    response = me.chat("Tell me about yourself", [])
    
    assert response == "Fallback Response"
    # Should still call LLM but without context
    mock_openai.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai.chat.completions.create.call_args
    messages = kwargs['messages']
    assert "## Contextual Evidence" not in messages[0]['content']

@patch('app.OpenAI')
@patch('app.RAGManager')
def test_me_chat_tool_calling_loop(mock_rag_class, mock_openai_class):
    """Test that the tool calling loop still works."""
    # Mock RAG
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []
    mock_rag.format_context.return_value = ""
    
    # Mock LLM to call a tool first, then stop
    mock_openai = mock_openai_class.return_value
    
    # Define tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "record_user_details"
    mock_tool_call.function.arguments = '{"email": "test@example.com"}'
    
    # First response: tool call
    resp1 = MagicMock()
    resp1.choices = [MagicMock(finish_reason="tool_calls", message=MagicMock(tool_calls=[mock_tool_call]))]
    
    # Second response: stop
    resp2 = MagicMock()
    resp2.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="Recorded!"))]
    
    mock_openai.chat.completions.create.side_effect = [resp1, resp2]
    
    with patch('app.record_user_details') as mock_tool:
        mock_tool.return_value = {"recorded": "ok"}
        me = Me()
        response = me.chat("contact me", [])
        
        assert response == "Recorded!"
        assert mock_tool.called
        assert mock_openai.chat.completions.create.call_count == 2
