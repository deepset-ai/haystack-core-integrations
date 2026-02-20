from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.dspy.chat.chat_generator import DSPyChatGenerator


@pytest.fixture
def mock_dspy_module():
    """
    Mock DSPy LM, configure, and module classes to avoid real API calls.
    """
    with patch("dspy.LM") as mock_lm_class, \
         patch("dspy.configure"), \
         patch("dspy.ChainOfThought") as mock_cot_class, \
         patch("dspy.Predict") as mock_predict_class, \
         patch("dspy.ReAct") as mock_react_class:
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        mock_module = MagicMock()
        mock_module.return_value = MagicMock(answer="Hello world!")
        mock_module.acall = AsyncMock(return_value=MagicMock(answer="Hello world!"))

        mock_cot_class.return_value = mock_module
        mock_predict_class.return_value = mock_module
        mock_react_class.return_value = mock_module

        yield mock_module


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class TestDSPyChatGeneratorAsync:
    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = await component.run_async(messages=chat_messages)

        # Verify the async mock was called
        mock_dspy_module.acall.assert_called_once()

        # Check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.asyncio
    async def test_run_async_with_params(self, chat_messages, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = await component.run_async(
            messages=chat_messages,
            generation_kwargs={"temperature": 0.9},
        )

        # Check that acall was called with the correct parameters
        _, kwargs = mock_dspy_module.acall.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.asyncio
    async def test_run_async_with_empty_messages(self, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        with pytest.raises(ValueError, match="messages"):
            await component.run_async(messages=[])
