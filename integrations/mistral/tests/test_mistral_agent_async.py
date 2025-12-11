# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haystack_integrations.components.agents.mistral.agent import MistralAgent


@pytest.mark.asyncio
class TestMistralAgentAsync:

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    async def test_run_async_basic(self, _mock_warm_up, chat_messages, mock_sdk_response, monkeypatch):
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
    async def test_run_async_empty_messages(self, _mock_warm_up, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test")
        result = await agent.run_async([])

        assert result == {"replies": []}
