from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import Message, TextBlockParam, Usage
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=lambda x: x,
    )
    return [tool]

def _canonical_completion() -> Message:
    return Message(
        id="foo",
        type="message",
        model="claude-sonnet-4-5",
        role="assistant",
        content=[TextBlockParam(type="text", text="Hello! I'm Claude.")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=57, output_tokens=40),
    )


@pytest.fixture
def mock_chat_completion():
    """Mock the synchronous Anthropic Messages.create and return the canonical reply."""
    with patch("anthropic.resources.messages.Messages.create") as mock_anthropic:
        mock_anthropic.return_value = _canonical_completion()
        yield mock_anthropic


@pytest.fixture
async def mock_anthropic_completion_async():
    """Mock the asynchronous Anthropic AsyncMessages.create and return the canonical reply."""
    with patch("anthropic.resources.messages.AsyncMessages.create") as mock_anthropic:
        # Make the mock return an awaitable
        mock_anthropic.return_value = AsyncMock(return_value=_canonical_completion())()
        yield mock_anthropic


@pytest.fixture
def mock_chat_completion_extended_thinking():
    """Mock the Anthropic Messages.create with an extended-thinking reply."""
    with patch("anthropic.resources.messages.Messages.create") as mock_chat_completion_create:
        completion = Message(
            id="foo",
            content=[
                {"type": "thinking", "thinking": "This is a thinking part!", "signature": ""},
                {"type": "text", "text": "Hello, world!"},
            ],
            model="claude-sonnet-4-5",
            role="assistant",
            type="message",
            usage={"input_tokens": 57, "output_tokens": 40},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"
