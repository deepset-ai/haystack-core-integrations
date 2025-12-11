# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret

from haystack_integrations.components.agents.mistral.agent import MistralAgent


class MockUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, call_id: str, name: str, arguments: str, call_type: str = "function"):
        self.id = call_id
        self.type = call_type
        self.function = MockFunction(name=name, arguments=arguments)


class MockMessage:
    """Mock message object from Mistral SDK response."""

    def __init__(self, content: str = "", tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(
        self, message: MockMessage, index: int = 0, finish_reason: str = "stop"
    ):
        self.message = message
        self.index = index
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(
        self, choices: list, model: str = "agent-model", usage: Optional[MockUsage] = None
    ):
        self.choices = choices
        self.model = model
        self.usage = usage or MockUsage()


class MockDelta:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls


class MockStreamChoice:
    def __init__(self, delta: MockDelta, index: int = 0, finish_reason: Optional[str] = None):
        self.delta = delta
        self.index = index
        self.finish_reason = finish_reason


class MockStreamData:
    """Mock streaming data object."""

    def __init__(self, choices: list, model: str = "agent-model", usage: Optional[MockUsage] = None):
        self.choices = choices
        self.model = model
        self.usage = usage


class MockStreamChunk:
    """Mock streaming chunk from SDK."""

    def __init__(self, data: MockStreamData):
        self.data = data


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]


@pytest.fixture
def mock_tool_function():
    def weather(city: str) -> str:
        return f"Weather in {city}: Sunny, 22°C"
    return weather


@pytest.fixture
def tools(mock_tool_function):
    return [
        Tool(
            name="weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            function=mock_tool_function,
        )
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


@pytest.fixture
def mock_sdk_response_with_tool_call():
    """Create a mock SDK response with tool calls."""
    tool_call = MockToolCall(
        call_id="call_123",
        name="weather",
        arguments='{"city": "Paris"}',
    )
    return MockResponse(
        choices=[
            MockChoice(
                message=MockMessage(content="", tool_calls=[tool_call]),
                finish_reason="tool_calls",
                index=0,
            )
        ],
        model="mistral-agent-model",
        usage=MockUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
    )


class TestMistralAgent:

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        agent = MistralAgent(agent_id="ag-test-123")

        assert agent.agent_id == "ag-test-123"
        assert agent.streaming_callback is None
        assert agent.tools is None
        assert agent.tool_choice is None
        assert agent.parallel_tool_calls is True
        assert agent.generation_kwargs == {}
        assert agent.timeout_ms == 30000

    def test_init_with_custom_parameters(self, tools):

        def my_callback(chunk):
            pass

        agent = MistralAgent(
            agent_id="ag-custom-456",
            api_key=Secret.from_token("custom-key"),
            streaming_callback=my_callback,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            generation_kwargs={"max_tokens": 500, "random_seed": 42},
            timeout_ms=60000,
        )

        assert agent.agent_id == "ag-custom-456"
        assert agent.streaming_callback is my_callback
        assert len(agent.tools) == 1
        assert agent.tool_choice == "auto"
        assert agent.parallel_tool_calls is False
        assert agent.generation_kwargs == {"max_tokens": 500, "random_seed": 42}
        assert agent.timeout_ms == 60000

    def test_init_with_duplicate_tools_raises_error(self, mock_tool_function):
        duplicate_tools = [
            Tool(
                name="weather",
                description="First",
                parameters={},
                function=mock_tool_function,
            ),
            Tool(
                name="weather",
                description="Second",
                parameters={},
                function=mock_tool_function,
            ),
        ]

        with pytest.raises(ValueError, match="Duplicate tool names"):
            MistralAgent(
                agent_id="ag-test",
                api_key=Secret.from_token("key"),
                tools=duplicate_tools,
            )

    def test_init_with_toolset(self, mock_tool_function, monkeypatch):
        """Test initialization with a Toolset."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters={},
            function=mock_tool_function,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters={},
            function=mock_tool_function,
        )
        toolset = Toolset([tool1, tool2])

        agent = MistralAgent(agent_id="ag-test", tools=[toolset])

        assert agent.tools == [toolset]

    def test_convert_simple_user_message(self, monkeypatch):
        """Test converting a simple user message."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        agent = MistralAgent(agent_id="ag-test")

        messages = [ChatMessage.from_user("Hello")]
        result = agent._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_system_message(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        agent = MistralAgent(agent_id="ag-test")

        messages = [ChatMessage.from_system("You are helpful")]
        result = agent._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"

    def test_convert_assistant_message(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        agent = MistralAgent(agent_id="ag-test")

        messages = [ChatMessage.from_assistant("I can help you")]
        result = agent._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I can help you"

    def test_convert_tool_message(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        agent = MistralAgent(agent_id="ag-test")

        tool_call = ToolCall(id="call_123", tool_name="weather", arguments={"city": "Paris"})
        messages = [ChatMessage.from_tool(tool_result="Sunny, 22°C", origin=tool_call)]
        result = agent._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Sunny, 22°C"
        assert result[0]["tool_call_id"] == "call_123"

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    def test_run_basic(
        self, chat_messages, mock_sdk_response, monkeypatch
    ):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test-123")
        agent._client = MagicMock()
        agent._client.agents.complete.return_value = mock_sdk_response

        result = agent.run(chat_messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert reply.text == "The capital of France is Paris."
        assert reply.meta["model"] == "mistral-agent-model"
        assert reply.meta["finish_reason"] == "stop"
        assert reply.meta["usage"]["total_tokens"] == 40

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    def test_run_empty_messages(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        agent = MistralAgent(agent_id="ag-test")
        result = agent.run([])
        assert result == {"replies": []}

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    def test_run_with_tool_calls(
        self,
        _mock_warm_up,
        mock_sdk_response_with_tool_call,
        tools,
        monkeypatch,
    ):

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test", tools=tools)
        agent._client = MagicMock()
        agent._client.agents.complete.return_value = mock_sdk_response_with_tool_call

        result = agent.run([ChatMessage.from_user("What's the weather in Paris?")])

        assert len(result["replies"]) == 1
        reply = result["replies"][0]

        assert reply.tool_calls is not None
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].tool_name == "weather"
        assert reply.tool_calls[0].arguments == {"city": "Paris"}
        assert reply.tool_calls[0].id == "call_123"
        assert reply.meta["finish_reason"] == "tool_calls"

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    def test_run_with_generation_kwargs(self,  chat_messages, mock_sdk_response, monkeypatch):

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(
            agent_id="ag-test", generation_kwargs={"max_tokens": 100}
        )
        agent._client = MagicMock()
        agent._client.agents.complete.return_value = mock_sdk_response

        # Run with additional kwargs
        agent.run(chat_messages, generation_kwargs={"random_seed": 42})

        # Verify the call was made with merged kwargs
        call_kwargs = agent._client.agents.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["random_seed"] == 42

    @patch(
        "haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up"
    )
    def test_run_with_tool_choice(self, chat_messages, mock_sdk_response, tools, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(agent_id="ag-test", tools=tools, tool_choice="any")
        agent._client = MagicMock()
        agent._client.agents.complete.return_value = mock_sdk_response

        agent.run(chat_messages)

        call_kwargs = agent._client.agents.complete.call_args.kwargs
        assert call_kwargs["tool_choice"] == "any"
        assert "tools" in call_kwargs
        assert call_kwargs["parallel_tool_calls"] is True

    @patch("haystack_integrations.components.agents.mistral.agent.MistralAgent.warm_up")
    def test_run_with_streaming(self, chat_messages, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        collected_chunks = []

        def callback(chunk: StreamingChunk):
            collected_chunks.append(chunk)

        # Create mock streaming response
        stream_chunks = [
            MockStreamChunk(
                data=MockStreamData(
                    choices=[MockStreamChoice(delta=MockDelta(content="The "))],
                    model="agent-model",
                )
            ),
            MockStreamChunk(
                data=MockStreamData(
                    choices=[MockStreamChoice(delta=MockDelta(content="capital "))],
                    model="agent-model",
                )
            ),
            MockStreamChunk(
                data=MockStreamData(
                    choices=[
                        MockStreamChoice(
                            delta=MockDelta(content="is Paris."), finish_reason="stop"
                        )
                    ],
                    model="agent-model",
                    usage=MockUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                )
            ),
        ]

        agent = MistralAgent(agent_id="ag-test", streaming_callback=callback)
        agent._client = MagicMock()
        agent._client.agents.stream.return_value = iter(stream_chunks)

        result = agent.run(chat_messages)

        # Verify streaming callback was called
        assert len(collected_chunks) == 3
        assert collected_chunks[0].content == "The "
        assert collected_chunks[1].content == "capital "
        assert collected_chunks[2].content == "is Paris."

        # Verify final message
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "The capital is Paris."
        assert result["replies"][0].meta["finish_reason"] == "stop"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        agent = MistralAgent(
            agent_id="ag-test",
            tool_choice="auto",
            generation_kwargs={"max_tokens": 500},
        )
        data = agent.to_dict()

        assert data["type"] == "haystack_integrations.components.agents.mistral.agent.MistralAgent"
        assert data["init_parameters"]["agent_id"] == "ag-test"
        assert data["init_parameters"]["tool_choice"] == "auto"
        assert data["init_parameters"]["generation_kwargs"] == {"max_tokens": 500}
        assert data["init_parameters"]["tools"] is None

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        data = {
            "type": "haystack_integrations.components.agents.mistral.agent.MistralAgent",
            "init_parameters": {
                "agent_id": "ag-restored",
                "api_key": {
                    "env_vars": ["MISTRAL_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "streaming_callback": None,
                "tools": None,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "generation_kwargs": {"max_tokens": 200},
                "timeout_ms": 45000,
            },
        }

        agent = MistralAgent.from_dict(data)

        assert agent.agent_id == "ag-restored"
        assert agent.tool_choice == "auto"
        assert agent.parallel_tool_calls is False
        assert agent.generation_kwargs == {"max_tokens": 200}
        assert agent.timeout_ms == 45000

    def test_pipeline_serialization(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        pipeline = Pipeline()
        pipeline.add_component("agent", MistralAgent(agent_id="ag-pipeline"))

        # Serialize to dict
        pipeline_dict = pipeline.to_dict()

        assert "agent" in pipeline_dict["components"]
        assert (
            pipeline_dict["components"]["agent"]["type"]
            == "haystack_integrations.components.agents.mistral.agent.MistralAgent"
        )

        # Deserialize
        restored_pipeline = Pipeline.from_dict(pipeline_dict)
        restored_agent = restored_pipeline.get_component("agent")

        assert restored_agent.agent_id == "ag-pipeline"

    def test_pipeline_yaml_roundtrip(self, monkeypatch):
        """Test YAML serialization and deserialization."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        pipeline = Pipeline()
        pipeline.add_component(
            "agent",
            MistralAgent(agent_id="ag-yaml", generation_kwargs={"max_tokens": 100}),
        )

        yaml_str = pipeline.dumps()
        restored = Pipeline.loads(yaml_str)

        agent = restored.get_component("agent")
        assert agent.agent_id == "ag-yaml"
        assert agent.generation_kwargs == {"max_tokens": 100}
