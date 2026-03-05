from unittest.mock import MagicMock, patch
from app import Me, TOOL_REGISTRY

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_unknown_tool_name(mock_rag_class, mock_get_llm):
    """Test that an unknown tool name is handled gracefully via TOOL_REGISTRY guard."""
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []

    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")

    # Stream requests a tool that doesn't exist in TOOL_REGISTRY
    def mock_stream_unknown_tool():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc = MagicMock()
        tc.index = 0; tc.id = "call_x"; tc.function.name = "hack_the_planet"; tc.function.arguments = '{}'
        chunk.choices[0].delta.tool_calls = [tc]
        yield chunk

    def mock_stream_final():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Done."
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    mock_llm_client.chat.completions.create.side_effect = [mock_stream_unknown_tool(), mock_stream_final()]

    me = Me()
    response = "".join(list(me.chat("do something weird", [])))

    # Should not crash — the TOOL_REGISTRY guard returns an error dict instead
    assert response == "Done."
    # Verify the tool result contains the error message
    args, kwargs = mock_llm_client.chat.completions.create.call_args
    messages = kwargs['messages']
    tool_msg = [m for m in messages if m.get('role') == 'tool'][0]
    assert "Tool 'hack_the_planet' not found or restricted" in tool_msg['content']

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_orchestration_success(mock_rag_class, mock_get_llm):
    """Test standard Me.chat orchestration with RAG context."""
    # Setup mocks
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = ["context chunk"]
    
    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")
    
    # Mock stream response
    def mock_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello!"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    mock_llm_client.chat.completions.create.return_value = mock_stream()
    
    me = Me()
    # chat() is a generator
    response_gen = me.chat("Tell me about yourself", [])
    response = "".join(list(response_gen))
    
    assert response == "Hello!"
    mock_rag.get_query_embedding.assert_called_once_with("Tell me about yourself")
    mock_llm_client.chat.completions.create.assert_called_once()
    
    # Verify system prompt contains context
    args, kwargs = mock_llm_client.chat.completions.create.call_args
    messages = kwargs['messages']
    assert messages[0]['role'] == 'system'
    assert "### CONTEXT ###" in messages[0]['content']
    assert "context chunk" in messages[0]['content']

@patch('app.get_llm')
@patch('app.RAGManager')
def test_chat_history_passing(mock_rag_class, mock_get_llm):
    """Test that conversation history is correctly passed to the LLM."""
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []

    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")

    def mock_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Response"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    mock_llm_client.chat.completions.create.return_value = mock_stream()

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    me = Me()
    list(me.chat("Another question", history))

    # Extract the messages sent to the LLM
    args, kwargs = mock_llm_client.chat.completions.create.call_args
    sent_messages = kwargs['messages']

    assert sent_messages[0]['role'] == 'system'
    assert sent_messages[1] == {"role": "user", "content": "Hi"}
    assert sent_messages[2] == {"role": "assistant", "content": "Hello"}
    assert sent_messages[3] == {"role": "user", "content": "Another question"}

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_tool_calling_loop(mock_rag_class, mock_get_llm):
    """Test that the tool calling loop works with streaming."""
    # Mock RAG
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []
    
    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")
    
    # Define tool call fragments
    def mock_stream_tool():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_1"
        tc_delta.function.name = "record_user_details"
        tc_delta.function.arguments = '{"email":'
        chunk.choices[0].delta.tool_calls = [tc_delta]
        yield chunk
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        tc_delta2 = MagicMock()
        tc_delta2.index = 0
        tc_delta2.id = None
        tc_delta2.function.name = ""
        tc_delta2.function.arguments = ' "test@example.com"}'
        chunk2.choices[0].delta.tool_calls = [tc_delta2]
        yield chunk2

    def mock_stream_final():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Recorded!"
        chunk.choices[0].delta.tool_calls = None
        yield chunk
    
    mock_llm_client.chat.completions.create.side_effect = [mock_stream_tool(), mock_stream_final()]
    
    # Patch the registry entry directly
    mock_tool = MagicMock(return_value={"recorded": "ok"})
    with patch.dict(TOOL_REGISTRY, {"record_user_details": mock_tool}):
        me = Me()
        response_gen = me.chat("contact me", [])
        response = "".join(list(response_gen))
        
        assert response == "Recorded!"
        assert mock_tool.called
        assert mock_llm_client.chat.completions.create.call_count == 2

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_max_turns_limit(mock_rag_class, mock_get_llm):
    """Test that the chat loop terminates after max_turns to prevent infinite loops."""
    # Mock RAG to return nothing
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []
    
    # Setup LLM client mock
    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")

    # Mock LLM to always return a tool call, never finishing
    def mock_infinite_tool():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc = MagicMock()
        tc.index = 0; tc.id = "call_inf"; tc.function.name = "record_unknown_question"; tc.function.arguments = '{"question": "why?"}'
        chunk.choices[0].delta.tool_calls = [tc]
        yield chunk

    mock_llm_client.chat.completions.create.side_effect = lambda **kwargs: mock_infinite_tool()

    me = Me()
    # Should run 10 turns and then exit the loop
    responses = list(me.chat("Infinite loop test", []))
    
    # 10 turns = 10 calls to completions.create
    assert mock_llm_client.chat.completions.create.call_count == 10

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_orchestration_fallback(mock_rag_class, mock_get_llm):
    """Test Me.chat fallback when RAG fails."""
    # Setup mocks
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.side_effect = Exception("RAG Error")
    
    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")
    
    # Mock stream response
    def mock_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Fallback Response"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    mock_llm_client.chat.completions.create.return_value = mock_stream()
    
    me = Me()
    response_gen = me.chat("Tell me about yourself", [])
    response = "".join(list(response_gen))
    
    assert response == "Fallback Response"
    mock_llm_client.chat.completions.create.assert_called_once()
    
    # Verify no context leaked into the system prompt
    args, kwargs = mock_llm_client.chat.completions.create.call_args
    messages = kwargs['messages']
    assert "### CONTEXT ###" not in messages[0]['content']
    assert "No specific portfolio context found" in messages[0]['content']

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_empty_response_fallback(mock_rag_class, mock_get_llm):
    """Test that an empty LLM response yields a user-friendly fallback message."""
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []

    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")

    # Mock stream that yields no content (empty delta)
    def mock_empty_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    mock_llm_client.chat.completions.create.return_value = mock_empty_stream()

    me = Me()
    response = "".join(list(me.chat("Hello?", [])))

    assert "couldn't generate a response" in response

@patch('app.get_llm')
@patch('app.RAGManager')
def test_me_chat_llm_exception_handling(mock_rag_class, mock_get_llm):
    """Test that an LLM crash yields a graceful error message instead of an unhandled exception."""
    mock_rag = mock_rag_class.return_value
    mock_rag.get_query_embedding.return_value = "vector"
    mock_rag.search.return_value = []

    mock_llm_client = MagicMock()
    mock_get_llm.return_value = (mock_llm_client, "test-model")

    # LLM raises an exception during streaming
    mock_llm_client.chat.completions.create.side_effect = Exception("API timeout")

    me = Me()
    response = "".join(list(me.chat("Tell me about yourself", [])))

    assert "trouble connecting" in response
