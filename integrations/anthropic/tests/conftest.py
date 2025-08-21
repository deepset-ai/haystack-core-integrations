from pathlib import Path
from unittest.mock import patch

import pytest
from anthropic.types import Message


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("anthropic.resources.messages.Messages.create") as mock_chat_completion_create:
        completion = Message(
            id="foo",
            content=[{"type": "text", "text": "Hello, world!"}],
            model="claude-sonnet-4-20250514",
            role="assistant",
            type="message",
            usage={"input_tokens": 57, "output_tokens": 40},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def mock_chat_completion_extended_thinking():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("anthropic.resources.messages.Messages.create") as mock_chat_completion_create:
        completion = Message(
            id="foo",
            content=[
                {"type": "thinking", "thinking": "This is a thinking part!", "signature": ""},
                {"type": "text", "text": "Hello, world!"},
            ],
            model="claude-sonnet-4-20250514",
            role="assistant",
            type="message",
            usage={"input_tokens": 57, "output_tokens": 40},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"
