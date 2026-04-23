# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest

from haystack_integrations.components.agents.ag2 import AG2Agent


class TestAG2AgentInit:
    """Test AG2Agent initialization."""

    def test_default_init(self):
        agent = AG2Agent()
        assert agent.model == "gpt-4o-mini"
        assert agent.system_message == "You are a helpful AI assistant."
        assert agent.api_key_env_var == "OPENAI_API_KEY"
        assert agent.api_type == "openai"
        assert agent.max_consecutive_auto_reply == 10
        assert agent.human_input_mode == "NEVER"
        assert agent.assistant_name == "assistant"
        assert agent.user_proxy_name == "user_proxy"
        assert agent.code_execution is False

    def test_custom_init(self):
        agent = AG2Agent(
            model="gpt-4o",
            system_message="Custom system message.",
            api_key_env_var="CUSTOM_API_KEY",
            api_type="openai",
            max_consecutive_auto_reply=5,
            human_input_mode="TERMINATE",
            assistant_name="researcher",
            user_proxy_name="coordinator",
            code_execution=True,
        )
        assert agent.model == "gpt-4o"
        assert agent.system_message == "Custom system message."
        assert agent.api_key_env_var == "CUSTOM_API_KEY"
        assert agent.max_consecutive_auto_reply == 5
        assert agent.human_input_mode == "TERMINATE"
        assert agent.assistant_name == "researcher"
        assert agent.user_proxy_name == "coordinator"
        assert agent.code_execution is True


class TestAG2AgentSerialization:
    """Test AG2Agent serialization/deserialization."""

    def test_to_dict(self):
        agent = AG2Agent(model="gpt-4o", system_message="Test message.")
        data = agent.to_dict()
        assert (
            data["type"] == "haystack_integrations.components.agents.ag2.agent.AG2Agent"
        )
        assert data["init_parameters"]["model"] == "gpt-4o"
        assert data["init_parameters"]["system_message"] == "Test message."

    def test_from_dict(self):
        agent = AG2Agent(model="gpt-4o", system_message="Test message.")
        data = agent.to_dict()
        restored = AG2Agent.from_dict(data)
        assert restored.model == "gpt-4o"
        assert restored.system_message == "Test message."

    def test_roundtrip_serialization(self):
        agent = AG2Agent(
            model="gpt-4o",
            system_message="Roundtrip test.",
            max_consecutive_auto_reply=5,
            code_execution=True,
        )
        data = agent.to_dict()
        restored = AG2Agent.from_dict(data)
        assert restored.model == agent.model
        assert restored.system_message == agent.system_message
        assert restored.max_consecutive_auto_reply == agent.max_consecutive_auto_reply
        assert restored.code_execution == agent.code_execution


class TestAG2AgentRun:
    """Test AG2Agent run method."""

    def test_run_missing_api_key(self):
        agent = AG2Agent(api_key_env_var="NONEXISTENT_KEY_12345")
        with pytest.raises(ValueError, match="Environment variable"):
            agent.run(query="test")

    @patch("haystack_integrations.components.agents.ag2.agent.UserProxyAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.AssistantAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.LLMConfig")
    def test_run_returns_reply(
        self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls
    ):
        os.environ["TEST_AG2_KEY"] = "test-key-123"
        try:
            # Setup mocks
            mock_assistant = MagicMock()
            mock_assistant_cls.return_value = mock_assistant
            mock_user_proxy = MagicMock()
            mock_user_proxy_cls.return_value = mock_user_proxy
            mock_assistant.chat_messages = {
                mock_user_proxy: [
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help? TERMINATE",
                    },
                ]
            }

            agent = AG2Agent(api_key_env_var="TEST_AG2_KEY")
            result = agent.run(query="Hello")

            assert "reply" in result
            assert "messages" in result
            assert result["reply"] == "Hello! How can I help?"
            mock_user_proxy.run.assert_called_once()
        finally:
            del os.environ["TEST_AG2_KEY"]

    @patch("haystack_integrations.components.agents.ag2.agent.UserProxyAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.AssistantAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.LLMConfig")
    def test_run_empty_messages(
        self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls
    ):
        os.environ["TEST_AG2_KEY"] = "test-key-123"
        try:
            mock_assistant = MagicMock()
            mock_assistant_cls.return_value = mock_assistant
            mock_user_proxy = MagicMock()
            mock_user_proxy_cls.return_value = mock_user_proxy
            mock_assistant.chat_messages = {mock_user_proxy: []}

            agent = AG2Agent(api_key_env_var="TEST_AG2_KEY")
            result = agent.run(query="Hello")

            assert result["reply"] == ""
            assert result["messages"] == []
        finally:
            del os.environ["TEST_AG2_KEY"]

    @patch("haystack_integrations.components.agents.ag2.agent.UserProxyAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.AssistantAgent")
    @patch("haystack_integrations.components.agents.ag2.agent.LLMConfig")
    def test_run_uses_correct_llm_config(
        self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls
    ):
        os.environ["TEST_AG2_KEY"] = "test-key-123"
        try:
            mock_assistant = MagicMock()
            mock_assistant_cls.return_value = mock_assistant
            mock_user_proxy = MagicMock()
            mock_user_proxy_cls.return_value = mock_user_proxy
            mock_assistant.chat_messages = {mock_user_proxy: []}

            agent = AG2Agent(
                model="gpt-4o",
                api_key_env_var="TEST_AG2_KEY",
                api_type="openai",
            )
            agent.run(query="test")

            # Verify LLMConfig was called with positional dict argument
            mock_llm_config.assert_called_once_with(
                {
                    "model": "gpt-4o",
                    "api_key": "test-key-123",
                    "api_type": "openai",
                }
            )
        finally:
            del os.environ["TEST_AG2_KEY"]
