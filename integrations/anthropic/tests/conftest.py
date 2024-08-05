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
            model="claude-3-sonnet-20240229",
            role="assistant",
            type="message",
            usage={"input_tokens": 57, "output_tokens": 40},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create
