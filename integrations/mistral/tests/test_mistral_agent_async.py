# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.agents.mistral.agent import MistralAgent


class MockUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMessage:
    def __init__(self, content: str = "", tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message: MockMessage, index: int = 0, finish_reason: str = "stop"):
        self.message = message
        self.index = index
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, choices: list, model: str = "agent-model", usage: Optional[MockUsage] = None):
        self.choices = choices
        self.model = model
        self.usage = usage or MockUsage()


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]


@pytest.fixture
def mock_sdk_response():
    return MockResponse(
        choices=[
            MockChoice(
                message=MockMessage(content="The capital of France is Paris."),
                finish_reason="stop",
                index=0,
            )
        ],
        model="mistral-agent-model",
        usage=MockUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
    )


@pytest.mark.asyncio
class TestMistralAgentAsync:

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    async def test_run_async_basic(self, chat_messages, mock_sdk_response, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test-123")
        agent._client = MagicMock()
        agent._client.agents.complete_async = AsyncMock(return_value=mock_sdk_response)

        result = await agent.run_async(chat_messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert reply.text == "The capital of France is Paris."
        assert reply.meta["model"] == "mistral-agent-model"
        assert reply.meta["finish_reason"] == "stop"
        assert reply.meta["usage"]["total_tokens"] == 40

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    async def test_run_async_empty_messages(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test")
        result = await agent.run_async([])

        assert result == {"replies": []}
