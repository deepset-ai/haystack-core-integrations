from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.dspy.chat.chat_generator import DSPySignatureChatGenerator


@pytest.fixture
def mock_dspy_module():
    """
    Mock DSPy LM, configure, and module classes to avoid real API calls.
    """
    with (
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.LM") as mock_lm_class,
        patch(
            "haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.ChainOfThought"
        ) as mock_cot_class,
        patch(
            "haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.Predict"
        ) as mock_predict_class,
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.ReAct") as mock_react_class,
    ):
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


class TestDSPySignatureChatGeneratorAsync:
    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        response = await component.run_async(messages=chat_messages)

        mock_dspy_module.acall.assert_called_once()

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.asyncio
    async def test_run_async_always_passes_config(self, chat_messages, mock_dspy_module):
        """Test that config is always passed (even as empty dict) in async mode."""
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        await component.run_async(messages=chat_messages)

        _, kwargs = mock_dspy_module.acall.call_args
        assert kwargs["config"] == {}

    @pytest.mark.asyncio
    async def test_run_async_with_params(self, chat_messages, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        response = await component.run_async(
            messages=chat_messages,
            generation_kwargs={"temperature": 0.9},
        )

        _, kwargs = mock_dspy_module.acall.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.asyncio
    async def test_run_async_with_empty_messages(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        with pytest.raises(ValueError, match="messages"):
            await component.run_async(messages=[])
