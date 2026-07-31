# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import Mock

import anthropic
import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    FileContent,
    ImageContent,
    StreamingChunk,
    TextContent,
    ToolCall,
)
from haystack.tools import Tool, Toolset, create_tool_from_function
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
    AnthropicChatGenerator,
)
from haystack_integrations.components.generators.anthropic.chat.utils import (
    _convert_messages_to_anthropic_format,
    _has_server_tool_blocks,
)


def hello_world():
    return "Hello, World!"


def population(city: str) -> str:
    return f"The population of {city} is 2.2 million"


class StreamingCollector:
    """Callable streaming callback that records chunks so live tests can assert on streaming behavior."""

    def __init__(self):
        self.responses = ""
        self.counter = 0

    def __call__(self, chunk: StreamingChunk) -> None:
        self.counter += 1
        self.responses += chunk.content if chunk.content else ""
        assert chunk.component_info is not None
        assert chunk.component_info.type.endswith("chat_generator.AnthropicChatGenerator")


@pytest.fixture
def tool_with_no_parameters():
    tool = Tool(
        name="hello_world",
        description="This prints hello world",
        parameters={"properties": {}, "type": "object"},
        function=hello_world,
    )
    return tool


class TestInit:
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = AnthropicChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        """
        Test the default initialization of the AnthropicChatGenerator component.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-sonnet-4-5"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None

    def test_init_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component initializes with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="claude-sonnet-4-5",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-sonnet-4-5"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == [tool]

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to initialize without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AnthropicChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        """
        Test that the AnthropicChatGenerator component fails to initialize with duplicate tool names.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            AnthropicChatGenerator(tools=duplicate_tools)

        # a server tool reusing a tool name would otherwise only fail as an opaque API error
        component = AnthropicChatGenerator(
            tools=tools,
            anthropic_server_tools=[{"type": "web_search_20250305", "name": tools[0].name}],
        )
        with pytest.raises(ValueError, match="Duplicate tool names"):
            component._prepare_request_params([ChatMessage.from_user("hi")])


class TestSerialization:
    def test_to_dict_default(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-5",
                "streaming_callback": None,
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {},
                "tools": None,
                "anthropic_server_tools": None,
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="claude-sonnet-4-5",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
            anthropic_server_tools=[{"type": "web_search_20250305", "name": "web_search"}],
            timeout=10.0,
            max_retries=1,
        )
        data = component.to_dict()

        # the Tool serialization format is owned by haystack-ai and varies across its versions; the
        # from_dict round-trip below covers the tools, so exclude them from the pinned-dict comparison
        tools_entries = data["init_parameters"].pop("tools")
        assert len(tools_entries) == 1

        expected_dict = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-5",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "anthropic_server_tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "timeout": 10.0,
                "max_retries": 1,
            },
        }

        assert data == expected_dict

        # deserializing the serialized component must reproduce the original tool
        loaded = AnthropicChatGenerator.from_dict(component.to_dict())
        assert loaded.tools == [tool]

    def test_from_dict(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be deserialized from a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-5",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {
                                "x": {
                                    "type": "string",
                                },
                            },
                        },
                    },
                ],
            },
        }
        component = AnthropicChatGenerator.from_dict(data)

        assert isinstance(component, AnthropicChatGenerator)
        assert component.model == "claude-sonnet-4-5"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("ANTHROPIC_API_KEY")
        assert component.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to deserialize from a dictionary without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-5",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError):
            AnthropicChatGenerator.from_dict(data)

    def test_serde_in_pipeline(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY", strict=False),
            model="claude-sonnet-4-5",
            generation_kwargs={"temperature": 0.6},
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()

        # the Tool serialization format is owned by haystack-ai and varies across its versions; the
        # dumps/loads round-trip below covers the tools, so exclude them from the pinned-dict comparison
        tools_entries = pipeline_dict["components"]["generator"]["init_parameters"].pop("tools")
        assert len(tools_entries) == 1
        type_ = "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator"

        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": type_,
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["ANTHROPIC_API_KEY"], "strict": False},
                        "model": "claude-sonnet-4-5",
                        "generation_kwargs": {"temperature": 0.6},
                        "ignore_tools_thinking_messages": True,
                        "streaming_callback": None,
                        "anthropic_server_tools": None,
                        "timeout": None,
                        "max_retries": None,
                    },
                }
            },
            "connections": [],
        }

        if not hasattr(pipeline, "_connection_type_validation"):
            expected_dict.pop("connection_type_validation")

        assert pipeline_dict == expected_dict

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline


class TestRun:
    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([ChatMessage.from_user("What's the capital of France?")], id="list"),
            pytest.param("What's the capital of France?", id="string"),
        ],
    )
    def test_run(self, mock_chat_completion, messages):
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = component.run(messages)

        # both a message list and a bare string must reach the backend as a single user message
        _, kwargs = mock_chat_completion.call_args
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"][0]["type"] == "text"
        assert kwargs["messages"][0]["content"][0]["text"] == "What's the capital of France?"

        # the result contains exactly one ChatMessage reply
        assert isinstance(result, dict)
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        """
        Test that the AnthropicChatGenerator component can run with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert "Hello! I'm Claude." in response["replies"][0].text
        assert response["replies"][0].meta["model"] == "claude-sonnet-4-5"
        assert response["replies"][0].meta["finish_reason"] == "stop"

    @pytest.mark.parametrize(
        "generation_kwargs,expected_kwargs",
        [
            (
                # a raw output_config is forwarded to the Anthropic API unchanged
                {
                    "output_config": {"effort": "medium"},
                },
                {
                    "output_config": {"effort": "medium"},
                },
            ),
            (
                {
                    "parallel_tool_use": False,
                    "tool_choice_type": "any",
                    "thinking_budget_tokens": 1024,
                },
                {
                    "tool_choice": {"disable_parallel_tool_use": True, "type": "any"},
                    "thinking": {"budget_tokens": 1024, "type": "enabled"},
                },
            ),
            (
                {
                    "parallel_tool_use": True,
                    "tool_choice_type": "all",
                },
                {
                    "tool_choice": {"disable_parallel_tool_use": False, "type": "all"},
                },
            ),
            (
                {
                    "parallel_tool_use": True,
                },
                {
                    "tool_choice": {"disable_parallel_tool_use": False, "type": "auto"},
                },
            ),
            (
                {
                    "disable_parallel_tool_use": True,
                },
                {
                    "tool_choice": {"disable_parallel_tool_use": True, "type": "auto"},
                },
            ),
            (
                {
                    "thinking_budget_tokens": None,
                    "parallel_tool_use": None,
                    "tool_choice_type": None,
                    "adaptive_thinking_effort": None,
                },
                {},
            ),
            (
                {
                    "adaptive_thinking_effort": "max",
                },
                {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "max"},
                },
            ),
            (
                {
                    "adaptive_thinking_effort": "disabled",
                },
                {
                    "thinking": {"type": "disabled"},
                },
            ),
            (
                {
                    "adaptive_thinking_effort": "none",
                },
                {
                    "thinking": {"type": "disabled"},
                },
            ),
        ],
    )
    def test_run_with_flattened_generation_kwargs(
        self, chat_messages, mock_chat_completion, generation_kwargs, expected_kwargs
    ):
        """
        Test that the AnthropicChatGenerator component can run with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs=generation_kwargs,
        )
        component.run(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        actual_kwargs = mock_chat_completion.call_args.kwargs
        assert actual_kwargs.get("tool_choice") == expected_kwargs.get("tool_choice")
        assert actual_kwargs.get("thinking") == expected_kwargs.get("thinking")
        assert actual_kwargs.get("output_config") == expected_kwargs.get("output_config")


class TestAnthropicServerTools:
    def test_init_with_anthropic_server_tools(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        server_tools = [{"type": "web_search_20250305"}]
        component = AnthropicChatGenerator(anthropic_server_tools=server_tools)
        assert component.anthropic_server_tools == server_tools

    def test_from_dict_with_anthropic_server_tools(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-5",
                "anthropic_server_tools": [{"type": "web_search_20250305"}],
            },
        }
        component = AnthropicChatGenerator.from_dict(data)
        assert component.anthropic_server_tools == [{"type": "web_search_20250305"}]

    def test_run_with_anthropic_server_tools(self, chat_messages, mock_chat_completion):
        server_tools = [{"type": "web_search_20250305"}]
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), anthropic_server_tools=server_tools
        )
        component.run(messages=chat_messages)
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["tools"] == server_tools

    def test_run_with_tools_and_anthropic_server_tools(self, chat_messages, mock_chat_completion, tools):
        server_tools = [{"type": "web_search_20250305"}]
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), tools=tools, anthropic_server_tools=server_tools
        )
        component.run(messages=chat_messages)
        _, kwargs = mock_chat_completion.call_args
        assert len(kwargs["tools"]) == 2
        assert kwargs["tools"][-1] == {"type": "web_search_20250305"}


class TestMixedToolsAndToolsets:
    def test_init_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test initialization with a mixed list of Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=lambda x: x)
        tool2 = Tool(
            name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=lambda y: y
        )
        tool3 = Tool(name="tool3", description="Third tool", parameters={"z": {"type": "string"}}, function=lambda z: z)
        toolset1 = Toolset([tool2])

        generator = AnthropicChatGenerator(tools=[tool1, toolset1, tool3])

        assert generator.tools == [tool1, toolset1, tool3]
        assert isinstance(generator.tools, list)
        assert len(generator.tools) == 3

    def test_init_with_duplicate_tools_in_mixed_list(self, monkeypatch):
        """Test that initialization fails with duplicate tool names in mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="duplicate", description="First", parameters={}, function=lambda: None)
        tool2 = Tool(name="duplicate", description="Second", parameters={}, function=lambda: None)
        toolset1 = Toolset([tool2])

        with pytest.raises(ValueError, match="duplicate"):
            AnthropicChatGenerator(tools=[tool1, toolset1])

    def test_serde_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test serialization/deserialization with mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=print)
        tool2 = Tool(name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=print)
        toolset1 = Toolset([tool2])

        generator = AnthropicChatGenerator(tools=[tool1, toolset1])
        data = generator.to_dict()

        # Verify serialization preserves structure
        assert isinstance(data["init_parameters"]["tools"], list)
        assert len(data["init_parameters"]["tools"]) == 2
        assert data["init_parameters"]["tools"][0]["type"] == "haystack.tools.tool.Tool"
        assert data["init_parameters"]["tools"][1]["type"] == "haystack.tools.toolset.Toolset"

        # Verify deserialization
        restored = AnthropicChatGenerator.from_dict(data)
        assert isinstance(restored.tools, list)
        assert len(restored.tools) == 2
        assert isinstance(restored.tools[0], Tool)
        assert isinstance(restored.tools[1], Toolset)
        assert restored.tools[0].name == "tool1"
        assert next(iter(restored.tools[1])).name == "tool2"

    def test_run_with_mixed_tools_and_toolsets(self, chat_messages, mock_chat_completion, monkeypatch):
        """Test that the run method works with mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=lambda x: x)
        tool2 = Tool(
            name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=lambda y: y
        )
        toolset1 = Toolset([tool2])

        component = AnthropicChatGenerator(tools=[tool1, toolset1])
        response = component.run(chat_messages)

        # Check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1

        # Check that the component called the Anthropic API with the correct tools
        _, kwargs = mock_chat_completion.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 2
        assert kwargs["tools"][0]["name"] == "tool1"
        assert kwargs["tools"][1]["name"] == "tool2"


class TestPromptCaching:
    def test_prompt_caching_enabled(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API when prompt
        caching is enabled
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        assert component.generation_kwargs.get("extra_headers", {}).get("anthropic-beta") == "prompt-caching-2024-07-31"

    def test_to_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly serialized to a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        data = component.to_dict()
        assert (
            data["init_parameters"]["generation_kwargs"]["extra_headers"]["anthropic-beta"]
            == "prompt-caching-2024-07-31"
        )

    def test_from_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly deserialized from a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-sonnet-4-5",
                "generation_kwargs": {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
            },
        }
        component = AnthropicChatGenerator.from_dict(data)
        assert component.generation_kwargs["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"

    @pytest.mark.parametrize("enable_caching", [True, False])
    def test_run_with_prompt_caching(self, monkeypatch, mock_chat_completion, enable_caching):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API in both cases of
        prompt caching being enabled or not
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if enable_caching else {}
        component = AnthropicChatGenerator(generation_kwargs=generation_kwargs)

        messages = [ChatMessage.from_system("System message"), ChatMessage.from_user("User message")]

        component.run(messages)

        # Check that the Anthropic API was called with the correct headers
        _, kwargs = mock_chat_completion.call_args
        headers = kwargs.get("extra_headers", {})
        if enable_caching:
            assert "anthropic-beta" in headers
        else:
            assert "anthropic-beta" not in headers

    def test_cache_control_forwarded_for_all_block_types(self, monkeypatch, mock_chat_completion):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()

        sys_msg = ChatMessage.from_system("sys")
        sys_msg._meta["cache_control"] = {"type": "ephemeral"}

        usr_msg = ChatMessage.from_user(
            "doc chunk",
            meta={"cache_control": {"type": "ephemeral"}},
        )

        tool_call = ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        asst_msg = ChatMessage.from_assistant(
            tool_calls=[tool_call],
            meta={"cache_control": {"type": "ephemeral"}},
        )

        tool_res = ChatMessage.from_tool(
            origin=tool_call,
            tool_result="sunny",
            meta={"cache_control": {"type": "ephemeral"}},
        )

        component.run([sys_msg, usr_msg, asst_msg, tool_res])

        _, kwargs = mock_chat_completion.call_args

        for blk in kwargs["system"]:
            assert blk.get("cache_control") == {"type": "ephemeral"}

        assert all("cache_control" not in msg for msg in kwargs["messages"])

        for msg in kwargs["messages"]:
            for cblk in msg["content"]:
                assert cblk.get("cache_control") == {"type": "ephemeral"}

    @pytest.mark.parametrize(
        "beta_header",
        [
            "featureA,extended-cache-ttl-2025-04-11",
            "featureA , featureB , new-fancy-stuff",
        ],
    )
    def test_extra_headers_pass_through(self, monkeypatch, mock_chat_completion, beta_header):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        component = AnthropicChatGenerator(generation_kwargs={"extra_headers": {"anthropic-beta": beta_header}})
        component.run([ChatMessage.from_user("ping")])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["extra_headers"]["anthropic-beta"] == beta_header

    def test_convert_messages_attaches_cache_control(self):
        user = ChatMessage.from_user(
            "hello",
            meta={
                "cache_control": {
                    "type": "ephemeral",
                }
            },
        )
        sys = ChatMessage.from_system("hi", meta={"cache_control": {"type": "ephemeral", "example_key": "example_val"}})
        sys_blocks, non_sys = _convert_messages_to_anthropic_format([sys, user])

        assert sys_blocks[0]["cache_control"] == {"type": "ephemeral", "example_key": "example_val"}
        assert non_sys[0]["content"][0]["cache_control"]["type"] == "ephemeral"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestIntegration:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run(self, streaming):
        """
        Integration test that the AnthropicChatGenerator component can run with default parameters,
        with and without streaming.
        """
        callback = StreamingCollector() if streaming else None
        component = AnthropicChatGenerator(streaming_callback=callback, timeout=30.0, max_retries=1)
        results = component.run(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-sonnet-4-5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        if streaming:
            assert callback.counter > 1
            assert "Paris" in callback.responses
            assert "input_tokens" in message.meta["usage"]
            assert "output_tokens" in message.meta["usage"]

    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicChatGenerator(model="something-obviously-wrong")
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    @pytest.mark.parametrize("tools_as_toolset", [False, True], ids=["list", "toolset"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_with_tools(self, tools, streaming, tools_as_toolset):
        """
        Integration test that the AnthropicChatGenerator component can run with tools passed either as a plain
        list or wrapped in a Toolset, with and without streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(
            tools=Toolset(tools) if tools_as_toolset else tools,
            streaming_callback=print_streaming_chunk if streaming else None,
            generation_kwargs={"max_tokens": 11000} if streaming else {},
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        if streaming:
            assert "output_tokens" in message.meta["usage"]
            assert "input_tokens" in message.meta["usage"]
        else:
            assert "completion_tokens" in message.meta["usage"]

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # the model tends to make tool calls if provided with tools, so we don't pass them here
        results = component.run(new_messages, generation_kwargs={"max_tokens": 50})

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    def test_live_run_with_parallel_tools(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with parallel tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AnthropicChatGenerator(tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # now we have the tool call
        assert len(message.tool_calls) == 2
        tool_call_paris = message.tool_calls[0]
        assert isinstance(tool_call_paris, ToolCall)
        assert tool_call_paris.id is not None
        assert tool_call_paris.tool_name == "weather"
        assert tool_call_paris.arguments["city"] in {"Paris", "Berlin"}
        assert message.meta["finish_reason"] == "tool_calls"

        tool_call_berlin = message.tool_calls[1]
        assert isinstance(tool_call_berlin, ToolCall)
        assert tool_call_berlin.id is not None
        assert tool_call_berlin.tool_name == "weather"
        assert tool_call_berlin.arguments["city"] in {"Berlin", "Paris"}

        # Anthropic expects results from both tools in the same message
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
        # [optional] Continue the conversation by sending a new message with the role of user, and a content block
        # containing the tool_result type and the following information:
        # tool_use_id: The id of the tool use request this is a result for.
        # content: The result of the tool, as a string (e.g. "content": "15 degrees") or list of
        # nested content blocks (e.g. "content": [{"type": "text", "text": "15 degrees"}]).
        # These content blocks can use the text or image types.
        # is_error (optional): Set to true if the tool execution resulted in an error.
        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call_paris, error=False),
            ChatMessage.from_tool(tool_result="12° C", origin=tool_call_berlin, error=False),
        ]

        # Response from the model contains results from both tools
        results = component.run(new_messages)
        message = results["replies"][0]
        assert not message.tool_calls
        assert len(message.text) > 0
        assert "paris" in message.text.lower()
        assert "berlin" in message.text.lower()
        assert "22°" in message.text
        assert "12°" in message.text
        assert message.meta["finish_reason"] == "stop"

    def test_live_run_with_tool_with_no_args_streaming(self, tool_with_no_parameters):
        """
        Integration test that the AnthropicChatGenerator component can run with a tool that has no arguments and
        streaming.
        """
        initial_messages = [ChatMessage.from_user("Print Hello World using the print hello world tool.")]
        component = AnthropicChatGenerator(tools=[tool_with_no_parameters], streaming_callback=print_streaming_chunk)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "hello_world"
        assert tool_call.arguments == {}
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="Hello World!", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "hello" in final_message.text.lower()

    def test_live_run_with_mixed_tools(self):
        """
        Integration test that verifies AnthropicChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """

        def weather(city: str):
            return f"The weather in {city} is sunny and 32°C"

        weather_tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for, e.g. Paris, London",
                    }
                },
                "required": ["city"],
            },
            function=weather,
        )

        population_tool = Tool(
            name="population",
            description="useful to determine the population of a given city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get population for, e.g. Paris, Berlin",
                    }
                },
                "required": ["city"],
            },
            function=population,
        )

        # Create a toolset with the population tool
        population_toolset = Toolset([population_tool])

        # Mix standalone tool with toolset
        mixed_tools = [weather_tool, population_toolset]

        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = AnthropicChatGenerator(tools=mixed_tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        first_reply = results["replies"][0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.tool_calls, "First reply has no tool calls"

        tool_calls = first_reply.tool_calls
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

        # Verify we got calls to both weather and population tools
        tool_names = {tc.tool_name for tc in tool_calls}
        assert "weather" in tool_names, "Expected 'weather' tool call"
        assert "population" in tool_names, "Expected 'population' tool call"

        # Verify tool call details
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name in ["weather", "population"]
            assert "city" in tool_call.arguments
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert first_reply.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, first_reply, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.parametrize("streaming_callback", [None, Mock()])
    def test_live_run_with_reasoning(self, streaming_callback):
        chat_generator = AnthropicChatGenerator(
            model="claude-sonnet-4-5",
            generation_kwargs={"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 11000},
            streaming_callback=streaming_callback,
        )

        message = ChatMessage.from_user("2+3?")
        response = chat_generator.run([message])["replies"][0]

        assert isinstance(response, ChatMessage)
        assert response.text and len(response.text) > 0
        assert response.reasoning is not None
        assert len(response.reasoning.reasoning_text) > 0

        new_message = ChatMessage.from_user("Now multiply the result by 10.")
        new_response = chat_generator.run([message, response, new_message])["replies"][0]
        assert isinstance(new_response, ChatMessage)
        assert new_response.text and len(new_response.text) > 0
        assert new_response.reasoning is not None
        assert len(new_response.reasoning.reasoning_text) > 0
        assert "reasoning_contents" in new_response.reasoning.extra

        if streaming_callback:
            streaming_callback.assert_called()

    def test_live_run_with_tools_streaming_and_reasoning(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools and streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(
            tools=tools,
            streaming_callback=print_streaming_chunk,
            generation_kwargs={
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "max_tokens": 11000,
            },
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # this is Anthropic thinking message prior to tool call
        assert message.reasoning.reasoning_text

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert "output_tokens" in message.meta["usage"]
        assert "input_tokens" in message.meta["usage"]

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.parametrize("cache_enabled", [True, False])
    @pytest.mark.parametrize("cache_location", ["system", "user"])
    def test_prompt_caching_live_run(self, cache_enabled, cache_location):
        """
        Prompt caching works whether the cache_control marker sits on the system or the user message.
        """
        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if cache_enabled else {}

        claude_llm = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"), generation_kwargs=generation_kwargs
        )

        # see https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-limitations
        if cache_location == "system":
            cached_message = ChatMessage.from_system(
                "This is the cached, here we make it at least 1024 tokens long." * 70
            )
            messages = [cached_message, ChatMessage.from_user("What's in cached content?")]
        else:
            cached_message = ChatMessage.from_user("This is a user message that should be long enough to cache. " * 100)
            messages = [ChatMessage.from_system("Hello from system. Just a generic instruction."), cached_message]

        if cache_enabled:
            cached_message._meta["cache_control"] = {"type": "ephemeral"}

        result = claude_llm.run(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        token_usage = result["replies"][0].meta.get("usage")

        if cache_enabled:
            # either we created cache or we read it (depends on how you execute this integration test)
            assert (
                token_usage.get("cache_creation_input_tokens", 0) > 1024
                or token_usage.get("cache_read_input_tokens", 0) > 1024
            ), f"Unexpected usage stats: {token_usage}"
        else:
            assert token_usage.get("cache_creation_input_tokens", 0) == 0
            assert token_usage.get("cache_read_input_tokens", 0) == 0

    def test_live_run_multimodal(self, test_files_path):
        """Integration test for multimodal functionality with real API."""
        image_path = test_files_path / "apple.jpg"
        # Resize the image to keep this test fast
        image_content = ImageContent.from_file_path(file_path=image_path, size=(100, 100))
        messages = [ChatMessage.from_user(content_parts=["What does this image show? Max 5 words", image_content])]

        generator = AnthropicChatGenerator(generation_kwargs={"max_tokens": 20})
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        message = response["replies"][0]
        assert message.text
        assert len(message.text) > 0
        assert any(word in message.text.lower() for word in ["apple", "fruit", "red"])

    def test_live_run_with_file_content(self, test_files_path):
        pdf_path = test_files_path / "sample_pdf_3.pdf"

        file_content = FileContent.from_file_path(
            file_path=pdf_path, extra={"context": "This document contains a table", "title": "A nice PDF"}
        )

        chat_messages = [
            ChatMessage.from_user(
                content_parts=[file_content, "Is this document a paper about LLMs? Respond with 'yes' or 'no' only."]
            )
        ]
        generator = AnthropicChatGenerator(model="claude-haiku-4-5")
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.is_from(ChatRole.ASSISTANT)

        assert message.text
        assert "no" in message.text.lower()

    def test_live_run_with_json_structured_output(self):
        """
        Integration test that the AnthropicChatGenerator component returns valid JSON
        when output_config.format with a json_schema is passed via generation_kwargs.
        """

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "plan_interest": {"type": "string"},
                "demo_requested": {"type": "boolean"},
            },
            "required": ["name", "email", "plan_interest", "demo_requested"],
            "additionalProperties": False,
        }

        component = AnthropicChatGenerator(
            generation_kwargs={
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }
        )
        results = component.run(
            messages=[
                ChatMessage.from_user(
                    "Extract the key information from this email: "
                    "John Smith (john@example.com) is interested in our Enterprise plan "
                    "and wants to schedule a demo for next Tuesday at 2pm."
                )
            ]
        )

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.meta["finish_reason"] == "stop"

        parsed = json.loads(message.text)
        assert parsed["name"] == "John Smith"
        assert parsed["email"] == "john@example.com"
        assert "enterprise" in parsed["plan_interest"].lower()
        assert parsed["demo_requested"] is True

    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_with_anthropic_server_tools(self, streaming):
        """The web search runs on Anthropic's side: it must never surface as a tool call to invoke."""
        component = AnthropicChatGenerator(
            anthropic_server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            streaming_callback=print_streaming_chunk if streaming else None,
        )
        results = component.run([ChatMessage.from_user("What is the Haystack framework by deepset? Search the web.")])

        message = results["replies"][0]
        assert "haystack" in message.text.lower()
        assert message.tool_calls == []
        assert message.meta["usage"]["server_tool_use"]["web_search_requests"] >= 1

        # raw blocks are kept so they can be replayed on the next turn
        raw_content = message.meta["raw_content_for_server_tools"]
        assert _has_server_tool_blocks(raw_content)
        search_results = next(b for b in raw_content if b["type"] == "web_search_tool_result")
        assert search_results["content"][0]["encrypted_content"]

        assert message.meta["citations"]
        assert message.meta["citations"][0]["url"]

        # the replayed turn carries the encrypted fields back unchanged
        _, non_system = _convert_messages_to_anthropic_format([message])
        assert non_system[0]["content"] == raw_content

    def test_live_run_agent_with_images_in_tool_result(self, test_files_path):
        def retrieve_image():
            return [
                TextContent("Here is the retrieved image."),
                ImageContent.from_file_path(test_files_path / "apple.jpg", size=(100, 100)),
            ]

        image_retriever_tool = create_tool_from_function(
            name="retrieve_image", description="Tool to retrieve an image", function=retrieve_image
        )
        image_retriever_tool.outputs_to_string = {"raw_result": True}

        agent = Agent(
            chat_generator=AnthropicChatGenerator(model="claude-haiku-4-5"),
            system_prompt="You are an Agent that can retrieve images and describe them.",
            tools=[image_retriever_tool],
        )

        user_message = ChatMessage.from_user("Retrieve the image and describe it in max 5 words.")
        result = agent.run(messages=[user_message])

        assert "apple" in result["last_message"].text.lower()

    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_agent_with_local_and_anthropic_server_tools(self, streaming):
        """An Agent must invoke only the client-side tool, while the server tool runs remotely."""
        invoked = []

        def unit_converter(km: float) -> str:
            invoked.append(km)
            return f"{km} km is {km * 0.621371:.2f} miles"

        agent = Agent(
            chat_generator=AnthropicChatGenerator(
                anthropic_server_tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                streaming_callback=print_streaming_chunk if streaming else None,
            ),
            tools=[
                Tool(
                    name="unit_converter",
                    description="Convert a distance in kilometers to miles.",
                    parameters={
                        "type": "object",
                        "properties": {"km": {"type": "number", "description": "distance in kilometers"}},
                        "required": ["km"],
                    },
                    function=unit_converter,
                )
            ],
            system_prompt="Research facts with web search. Always use unit_converter for km->miles.",
        )
        agent.warm_up()

        result = agent.run(
            messages=[
                ChatMessage.from_user(
                    "Search the web for the distance from Berlin to Rome in kilometers, "
                    "then use the unit_converter tool to convert it to miles."
                )
            ]
        )

        # the client-side tool ran locally...
        assert invoked
        # ...and no server tool ever leaked into tool_calls (an empty name would raise in ToolInvoker)
        called = [tc.tool_name for m in result["messages"] for tc in (m.tool_calls or [])]
        assert called == ["unit_converter"]

        searched = any(
            (m.meta.get("usage") or {}).get("server_tool_use", {}).get("web_search_requests")
            for m in result["messages"]
        )
        assert searched
