# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline

from haystack_integrations.components.agents.ag2.ag2_agent import (
    HUMAN_INPUT_MODES,
    AG2Agent,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_chat_result(messages: list[dict]) -> MagicMock:
    """Build a mock chat result with the given message history."""
    result = MagicMock()
    result.chat_history = messages
    return result


@pytest.fixture()
def mock_ag2():
    """
    Patch AssistantAgent and UserProxyAgent so no real AG2/network calls happen.

    Returns (mock_assistant_cls, mock_user_proxy_cls, mock_user_proxy_instance).
    """
    mock_assistant_cls = MagicMock()
    mock_assistant_instance = MagicMock()
    mock_assistant_cls.return_value = mock_assistant_instance

    mock_user_proxy_cls = MagicMock()
    mock_user_proxy_instance = MagicMock()
    mock_user_proxy_cls.return_value = mock_user_proxy_instance

    default_chat_result = _make_chat_result(
        [
            {"role": "assistant", "content": "Hello", "name": "user_proxy"},
            {"role": "user", "content": "Hi there! How can I help?", "name": "assistant"},
        ]
    )
    mock_user_proxy_instance.initiate_chat.return_value = default_chat_result

    with (
        patch(
            "haystack_integrations.components.agents.ag2.ag2_agent.AssistantAgent",
            mock_assistant_cls,
        ),
        patch(
            "haystack_integrations.components.agents.ag2.ag2_agent.UserProxyAgent",
            mock_user_proxy_cls,
        ),
    ):
        yield mock_assistant_cls, mock_user_proxy_cls, mock_user_proxy_instance


# ---------------------------------------------------------------------------
# HUMAN_INPUT_MODES constant
# ---------------------------------------------------------------------------


class TestHumanInputModes:
    def test_contains_expected_modes(self):
        assert HUMAN_INPUT_MODES == {"ALWAYS", "NEVER", "TERMINATE"}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestAG2AgentInit:
    def test_defaults(self):
        agent = AG2Agent()
        assert agent.model == "gpt-4o-mini"
        assert agent.api_type == "openai"
        assert agent.system_message is None
        assert agent.human_input_mode == "NEVER"
        assert agent.code_execution is False
        assert agent.max_consecutive_auto_reply == 10

    def test_custom_params(self):
        agent = AG2Agent(
            model="gpt-4o",
            api_type="azure",
            system_message="You are a coder.",
            human_input_mode="TERMINATE",
            code_execution=True,
            max_consecutive_auto_reply=5,
        )
        assert agent.model == "gpt-4o"
        assert agent.api_type == "azure"
        assert agent.system_message == "You are a coder."
        assert agent.human_input_mode == "TERMINATE"
        assert agent.code_execution is True
        assert agent.max_consecutive_auto_reply == 5

    def test_invalid_human_input_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid human_input_mode"):
            AG2Agent(human_input_mode="INVALID")

    def test_invalid_human_input_mode_lists_valid(self):
        with pytest.raises(ValueError, match="NEVER"):
            AG2Agent(human_input_mode="WRONG")

    @pytest.mark.parametrize("mode", ["ALWAYS", "NEVER", "TERMINATE"])
    def test_all_valid_modes_accepted(self, mode):
        agent = AG2Agent(human_input_mode=mode)
        assert agent.human_input_mode == mode


# ---------------------------------------------------------------------------
# _build_llm_config
# ---------------------------------------------------------------------------


class TestBuildLLMConfig:
    def test_default(self):
        agent = AG2Agent()
        cfg = agent._build_llm_config()
        assert cfg == {"config_list": [{"model": "gpt-4o-mini", "api_type": "openai"}]}

    def test_custom_model_and_api_type(self):
        agent = AG2Agent(model="gpt-4o", api_type="azure")
        cfg = agent._build_llm_config()
        assert cfg == {"config_list": [{"model": "gpt-4o", "api_type": "azure"}]}


# ---------------------------------------------------------------------------
# _build_code_execution_config
# ---------------------------------------------------------------------------


class TestBuildCodeExecutionConfig:
    def test_code_execution_off_returns_false(self):
        agent = AG2Agent(code_execution=False)
        assert agent._build_code_execution_config() is False

    def test_code_execution_on_returns_dict(self):
        agent = AG2Agent(code_execution=True)
        result = agent._build_code_execution_config()
        assert isinstance(result, dict)
        assert result.get("use_docker") is False


# ---------------------------------------------------------------------------
# _extract_last_assistant_reply
# ---------------------------------------------------------------------------


class TestExtractLastAssistantReply:
    def test_returns_last_assistant_message(self):
        # AG2 format: user_proxy messages have role="assistant", assistant messages have role="user"
        # We identify the real assistant by name="assistant"
        messages = [
            {"role": "assistant", "content": "Hello", "name": "user_proxy"},
            {"role": "user", "content": "First reply", "name": "assistant"},
            {"role": "assistant", "content": "Follow-up", "name": "user_proxy"},
            {"role": "user", "content": "Second reply", "name": "assistant"},
        ]
        assert AG2Agent._extract_last_assistant_reply(messages) == "Second reply"

    def test_skips_empty_content(self):
        messages = [
            {"role": "user", "content": "", "name": "assistant"},
            {"role": "user", "content": "Real reply", "name": "assistant"},
        ]
        assert AG2Agent._extract_last_assistant_reply(messages) == "Real reply"

    def test_raises_when_no_assistant_message(self):
        messages = [{"role": "assistant", "content": "Hello", "name": "user_proxy"}]
        with pytest.raises(ValueError, match="No assistant reply"):
            AG2Agent._extract_last_assistant_reply(messages)

    def test_raises_on_empty_history(self):
        with pytest.raises(ValueError, match="No assistant reply"):
            AG2Agent._extract_last_assistant_reply([])

    def test_none_content_is_skipped(self):
        messages = [
            {"role": "user", "content": None, "name": "assistant"},
            {"role": "user", "content": "Valid reply", "name": "assistant"},
        ]
        assert AG2Agent._extract_last_assistant_reply(messages) == "Valid reply"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestAG2AgentRun:
    def test_run_returns_reply_and_messages(self, mock_ag2):
        _, _, _mock_proxy = mock_ag2
        agent = AG2Agent()
        result = agent.run(query="What is RAG?")

        assert "reply" in result
        assert "messages" in result
        assert result["reply"] == "Hi there! How can I help?"
        assert len(result["messages"]) == 2

    def test_run_calls_initiate_chat_with_query(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        agent = AG2Agent()
        agent.run(query="Explain transformers.")

        mock_proxy.initiate_chat.assert_called_once()
        call_kwargs = mock_proxy.initiate_chat.call_args
        assert call_kwargs.kwargs.get("message") == "Explain transformers." or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "Explain transformers."
        )

    def test_run_passes_silent_true(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        agent = AG2Agent()
        agent.run(query="Hello")

        call_kwargs = mock_proxy.initiate_chat.call_args.kwargs
        assert call_kwargs.get("silent") is True

    def test_run_creates_assistant_with_llm_config(self, mock_ag2):
        mock_assistant_cls, _, _ = mock_ag2
        agent = AG2Agent(model="gpt-4o", api_type="openai")
        agent.run(query="Hello")

        call_kwargs = mock_assistant_cls.call_args.kwargs
        assert call_kwargs["llm_config"] == {"config_list": [{"model": "gpt-4o", "api_type": "openai"}]}

    def test_run_passes_system_message_when_set(self, mock_ag2):
        mock_assistant_cls, _, _ = mock_ag2
        agent = AG2Agent(system_message="You are a poet.")
        agent.run(query="Write a poem.")

        call_kwargs = mock_assistant_cls.call_args.kwargs
        assert call_kwargs.get("system_message") == "You are a poet."

    def test_run_omits_system_message_when_none(self, mock_ag2):
        mock_assistant_cls, _, _ = mock_ag2
        agent = AG2Agent(system_message=None)
        agent.run(query="Hello")

        call_kwargs = mock_assistant_cls.call_args.kwargs
        assert "system_message" not in call_kwargs

    def test_run_creates_user_proxy_without_llm(self, mock_ag2):
        _, mock_proxy_cls, _ = mock_ag2
        agent = AG2Agent()
        agent.run(query="Hello")

        call_kwargs = mock_proxy_cls.call_args.kwargs
        assert call_kwargs.get("llm_config") is False

    def test_run_code_execution_disabled(self, mock_ag2):
        _, mock_proxy_cls, _ = mock_ag2
        agent = AG2Agent(code_execution=False)
        agent.run(query="Hello")

        call_kwargs = mock_proxy_cls.call_args.kwargs
        assert call_kwargs.get("code_execution_config") is False

    def test_run_code_execution_enabled(self, mock_ag2):
        _, mock_proxy_cls, _ = mock_ag2
        agent = AG2Agent(code_execution=True)
        agent.run(query="Hello")

        call_kwargs = mock_proxy_cls.call_args.kwargs
        assert call_kwargs.get("code_execution_config") == {"use_docker": False}

    def test_run_passes_max_consecutive_auto_reply(self, mock_ag2):
        _, mock_proxy_cls, _ = mock_ag2
        agent = AG2Agent(max_consecutive_auto_reply=3)
        agent.run(query="Hello")

        call_kwargs = mock_proxy_cls.call_args.kwargs
        assert call_kwargs.get("max_consecutive_auto_reply") == 3

    def test_run_raises_when_no_assistant_reply(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        mock_proxy.initiate_chat.return_value = _make_chat_result(
            [{"role": "assistant", "content": "Hello", "name": "user_proxy"}]
        )
        agent = AG2Agent()
        with pytest.raises(ValueError, match="No assistant reply"):
            agent.run(query="Hello")

    def test_run_handles_empty_history(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        mock_proxy.initiate_chat.return_value = _make_chat_result([])
        agent = AG2Agent()
        with pytest.raises(ValueError, match="No assistant reply"):
            agent.run(query="Hello")

    def test_run_handles_none_chat_history(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        result = MagicMock()
        result.chat_history = None
        mock_proxy.initiate_chat.return_value = result
        agent = AG2Agent()
        with pytest.raises(ValueError, match="No assistant reply"):
            agent.run(query="Hello")

    def test_run_multi_turn_picks_last_assistant(self, mock_ag2):
        _, _, mock_proxy = mock_ag2
        mock_proxy.initiate_chat.return_value = _make_chat_result(
            [
                {"role": "assistant", "content": "Hi", "name": "user_proxy"},
                {"role": "user", "content": "First answer", "name": "assistant"},
                {"role": "assistant", "content": "More info?", "name": "user_proxy"},
                {"role": "user", "content": "Detailed answer", "name": "assistant"},
            ]
        )
        agent = AG2Agent()
        result = agent.run(query="Hi")
        assert result["reply"] == "Detailed answer"

    def test_run_human_input_mode_passed_to_proxy(self, mock_ag2):
        _, mock_proxy_cls, _ = mock_ag2
        agent = AG2Agent(human_input_mode="TERMINATE")
        agent.run(query="Hello")

        call_kwargs = mock_proxy_cls.call_args.kwargs
        assert call_kwargs.get("human_input_mode") == "TERMINATE"


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_defaults(self):
        agent = AG2Agent()
        data = agent.to_dict()
        assert data == {
            "type": "haystack_integrations.components.agents.ag2.ag2_agent.AG2Agent",
            "init_parameters": {
                "model": "gpt-4o-mini",
                "api_type": "openai",
                "system_message": None,
                "human_input_mode": "NEVER",
                "code_execution": False,
                "max_consecutive_auto_reply": 10,
            },
        }

    def test_to_dict_with_all_params(self):
        agent = AG2Agent(
            model="gpt-4o",
            api_type="azure",
            system_message="Be concise.",
            human_input_mode="TERMINATE",
            code_execution=True,
            max_consecutive_auto_reply=5,
        )
        data = agent.to_dict()
        params = data["init_parameters"]
        assert params["model"] == "gpt-4o"
        assert params["api_type"] == "azure"
        assert params["system_message"] == "Be concise."
        assert params["human_input_mode"] == "TERMINATE"
        assert params["code_execution"] is True
        assert params["max_consecutive_auto_reply"] == 5

    def test_from_dict_defaults(self):
        data = {
            "type": "haystack_integrations.components.agents.ag2.ag2_agent.AG2Agent",
            "init_parameters": {
                "model": "gpt-4o-mini",
                "api_type": "openai",
                "system_message": None,
                "human_input_mode": "NEVER",
                "code_execution": False,
                "max_consecutive_auto_reply": 10,
            },
        }
        agent = AG2Agent.from_dict(data)
        assert agent.model == "gpt-4o-mini"
        assert agent.api_type == "openai"
        assert agent.system_message is None
        assert agent.human_input_mode == "NEVER"
        assert agent.code_execution is False
        assert agent.max_consecutive_auto_reply == 10

    def test_from_dict_with_all_params(self):
        data = {
            "type": "haystack_integrations.components.agents.ag2.ag2_agent.AG2Agent",
            "init_parameters": {
                "model": "gpt-4o",
                "api_type": "azure",
                "system_message": "Be concise.",
                "human_input_mode": "TERMINATE",
                "code_execution": True,
                "max_consecutive_auto_reply": 5,
            },
        }
        agent = AG2Agent.from_dict(data)
        assert agent.model == "gpt-4o"
        assert agent.api_type == "azure"
        assert agent.system_message == "Be concise."
        assert agent.human_input_mode == "TERMINATE"
        assert agent.code_execution is True
        assert agent.max_consecutive_auto_reply == 5

    def test_roundtrip(self):
        original = AG2Agent(
            model="gpt-4o",
            api_type="azure",
            system_message="You are helpful.",
            human_input_mode="TERMINATE",
            code_execution=True,
            max_consecutive_auto_reply=3,
        )
        restored = AG2Agent.from_dict(original.to_dict())
        assert restored.model == original.model
        assert restored.api_type == original.api_type
        assert restored.system_message == original.system_message
        assert restored.human_input_mode == original.human_input_mode
        assert restored.code_execution == original.code_execution
        assert restored.max_consecutive_auto_reply == original.max_consecutive_auto_reply

    def test_from_dict_invalid_mode_raises(self):
        data = {
            "type": "haystack_integrations.components.agents.ag2.ag2_agent.AG2Agent",
            "init_parameters": {
                "model": "gpt-4o-mini",
                "api_type": "openai",
                "system_message": None,
                "human_input_mode": "INVALID",
                "code_execution": False,
                "max_consecutive_auto_reply": 10,
            },
        }
        with pytest.raises(ValueError, match="Invalid human_input_mode"):
            AG2Agent.from_dict(data)


# ---------------------------------------------------------------------------
# Haystack pipeline integration (no real AG2 calls)
# ---------------------------------------------------------------------------


class TestHaystackPipelineIntegration:
    def test_component_in_pipeline(self, mock_ag2):
        pipeline = Pipeline()
        pipeline.add_component("agent", AG2Agent())
        result = pipeline.run({"agent": {"query": "What is RAG?"}})

        assert "agent" in result
        assert "reply" in result["agent"]
        assert "messages" in result["agent"]

    def test_component_output_types(self):
        agent = AG2Agent()
        run_method = agent.run
        output_types = getattr(run_method, "__haystack_output_types__", None)
        if output_types is None:
            assert callable(run_method)
        else:
            assert "reply" in output_types
            assert "messages" in output_types


# ---------------------------------------------------------------------------
# Integration tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Export OPENAI_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestAG2AgentIntegration:
    def test_live_run(self):
        agent = AG2Agent(model="gpt-4o-mini", max_consecutive_auto_reply=2)
        result = agent.run(query="What is 2 + 2? Reply with just the number.")
        assert isinstance(result["reply"], str)
        assert "4" in result["reply"]
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0

    def test_live_run_with_system_message(self):
        agent = AG2Agent(
            model="gpt-4o-mini",
            system_message="Always answer in exactly one word.",
            max_consecutive_auto_reply=2,
        )
        result = agent.run(query="What is the capital of France?")
        assert "Paris" in result["reply"]

    def test_live_roundtrip_serialization(self):
        original = AG2Agent(model="gpt-4o-mini", system_message="Be brief.", max_consecutive_auto_reply=2)
        restored = AG2Agent.from_dict(original.to_dict())
        result = restored.run(query="Say 'hello'.")
        assert isinstance(result["reply"], str)
