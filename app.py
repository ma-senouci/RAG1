import sys
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from rag_logic import RAGManager, logger
from llm_factory import get_llm

load_dotenv(override=True)

def push(text):
    """Send a push notification via Pushover."""
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if token and user:
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": token,
                    "user": user,
                    "message": text,
                },
                timeout=5
            )
        except Exception as e:
            logger.warning(f"Failed to send push notification: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Tool to record user interest and contact info."""
    push(f"User Interest: {name} ({email}) - Notes: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Tool to record questions the AI couldn't answer."""
    push(f"Unknown Question: {question}")
    return {"recorded": "ok"}

# Tool Registry for security (prevents arbitrary function execution)
TOOL_REGISTRY = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}

# Tool definitions for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address of this user"},
                    "name": {"type": "string", "description": "The user's name, if they provided it"},
                    "notes": {"type": "string", "description": "Context about the conversation"}
                },
                "required": ["email"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered due to lack of information",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The unanswered question"}
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    }
]

class Me:
    """Class representing the user's persona with RAG and Tool-calling capabilities."""

    def __init__(self):
        self.name = "Mohamed Abdelkrim SENOUCI"
        # Factory-based LLM instantiation (defaulting to DeepSeek via environment)
        try:
            self.llm_client, self.model_name = get_llm()
            logger.info(f"LLM initialized: {self.model_name} (via {os.getenv('LLM_BACKEND', 'deepseek')})")

            # Initialize RAG
            self.rag = RAGManager()
        except Exception as e:
            logger.error(f"Configuration or Initialization Error: {e}")
            logger.error("Please check your .env file.")
            sys.exit(1)

    def format_context(self, context_chunks: list[str]) -> str:
        """
        Formats retrieved context chunks for system prompt injection.
        """
        if not context_chunks:
            return ""
        
        header = f"Here is relevant context from {self.name}'s portfolio documents:\n"
        joined_chunks = "\n\n".join(context_chunks)
        
        return f"{header}{joined_chunks}"

    def system_prompt(self, context=""):
        """Generate the system prompt with optional context injection."""
        prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of background and CV which you can use to answer questions. "
            "Be professional and engaging, as if talking to a potential client or future employer. "
            "If you don't know the answer to any question, use your record_unknown_question tool to record it. "
            "If the user is engaging in discussion, try to steer them towards getting in touch; "
            "ask for their email and record it using your record_user_details tool."
        )

        if context:
            prompt += f"\n\n### CONTEXT ###\n{context}\n###############\n"
        else:
            prompt += f"\n\n(No specific portfolio context found for this query. Use your general knowledge of {self.name}'s background if possible, or gracefully record any unknowns.)"

        prompt += f"\n\nPlease chat with the user, ALWAYS staying in character as {self.name}."
        return prompt

    def handle_tool_call(self, tool_calls):
        """Execute tool calls and return results for the LLM."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info(f"Tool called: {tool_name}")

            # Safe lookup via registry
            tool_func = TOOL_REGISTRY.get(tool_name)
            result = tool_func(**arguments) if tool_func else {"error": f"Tool '{tool_name}' not found or restricted"}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results
    
    def chat(self, message, history):
        """Orchestrate RAG retrieval and streaming LLM response."""
        logger.info(f"Processing message: {message[:50]}...")

        # 1. Retrieve relevant portfolio chunks via semantic search
        try:
            query_vector = self.rag.get_query_embedding(message)
            context_chunks = self.rag.search(query_vector, k=3)
            formatted_context = self.format_context(context_chunks)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            formatted_context = ""

        # 2. Augment prompt with retrieved context, persona, and conversation history
        system_content = self.system_prompt(context=formatted_context)
        messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]

        # 3. Generate response via LLM, resolving any tool calls before final reply
        try:
            full_response = ""
            max_turns = 10
            turns = 0

            while turns < max_turns:
                stream = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    stream=True
                )

                tool_calls = []
                content_yielded = False

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Handle content streaming
                    if delta.content:
                        full_response += delta.content
                        content_yielded = True
                        yield full_response

                    # Buffer tool calls (reassemble from fragments)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            if len(tool_calls) <= tc_delta.index:
                                tool_calls.append({
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })

                            if tc_delta.id:
                                tool_calls[tc_delta.index]["id"] = tc_delta.id
                            if tc_delta.function.name:
                                tool_calls[tc_delta.index]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments

                if tool_calls:
                    # Compatibility bridge: wrap dicts into Mock objects for handle_tool_call
                    class MockFunction:
                        def __init__(self, name, arguments):
                            self.name = name
                            self.arguments = arguments
                    class MockToolCall:
                        def __init__(self, id, function):
                            self.id = id
                            self.function = function

                    formatted_tool_calls = [
                        MockToolCall(tc["id"], MockFunction(tc["function"]["name"], tc["function"]["arguments"]))
                        for tc in tool_calls
                    ]

                    tool_results = self.handle_tool_call(formatted_tool_calls)

                    # Append the assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            } for tc in tool_calls
                        ]
                    })
                    messages.extend(tool_results)
                    # Loop continues for next turn
                else:
                    # No tool calls, we are done
                    break

                turns += 1

            if not full_response and not content_yielded:
                 yield "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            yield "I apologize, but I'm having trouble connecting to my brain right now. Please try again in a moment."


if __name__ == "__main__":
    me = Me()

    custom_css = """
    .gradio-container { max-width: 850px !important; margin: auto !important; }
    h1 { color: #0d9488 !important; font-weight: 700 !important; text-align: center !important; }
    .description { color: #64748b !important; font-size: 1.05rem !important; text-align: center !important; }
    """

    interface = gr.ChatInterface(
        fn=me.chat,
        title=f"💬 Chat with {me.name}",
        description="Ask me about my experience, projects, or background — answers grounded in real portfolio data.",
        textbox=gr.Textbox(placeholder="Type your question here...", submit_btn=True),
        examples=[
            "Tell me about your experience with AI.",
            "What projects have you worked on?",
            "How can I contact you?"
        ],
    )
    interface.launch(
        theme=gr.themes.Base(
                primary_hue="teal",
                secondary_hue="slate",
                font=gr.themes.GoogleFont("Inter")
            ),
        css=custom_css,
    )
