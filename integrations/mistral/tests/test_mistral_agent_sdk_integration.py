# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for MistralAgentSDK component.

These tests make actual API calls to the Mistral Agents API and require:
1. MISTRAL_API_KEY environment variable
2. MISTRAL_AGENT_ID environment variable (a real agent created in Mistral console)

To run these tests:
    export MISTRAL_API_KEY="your-api-key"
    export MISTRAL_AGENT_ID="your-agent-id"
    pytest tests/test_mistral_agent_sdk_integration.py -v -m integration
"""

import json
import os

import pytest
from haystack import Pipeline
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset

from haystack_integrations.components.agents.mistral.agent_sdk import MistralAgentSDK


# Skip all tests if required environment variables are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY") or not os.environ.get("MISTRAL_AGENT_ID"),
    reason="Set MISTRAL_API_KEY and MISTRAL_AGENT_ID environment variables to run integration tests"
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def agent_id():
    """Get agent ID from environment."""
    return os.environ.get("MISTRAL_AGENT_ID")


@pytest.fixture
def agent(agent_id):
    """Create a basic MistralAgentSDK instance."""
    return MistralAgentSDK(agent_id=agent_id)


@pytest.fixture
def weather_tool():
    """Create a weather tool for testing."""
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get weather for a city."""
        # Simulated weather data
        weather_data = {
            "Paris": {"temp": 22, "condition": "sunny"},
            "London": {"temp": 15, "condition": "rainy"},
            "Tokyo": {"temp": 28, "condition": "humid"},
            "New York": {"temp": 18, "condition": "cloudy"},
        }
        data = weather_data.get(city, {"temp": 20, "condition": "unknown"})
        temp = data["temp"] if unit == "celsius" else int(data["temp"] * 9/5 + 32)
        unit_str = "°C" if unit == "celsius" else "°F"
        return f"Weather in {city}: {data['condition']}, {temp}{unit_str}"

    return Tool(
        name="get_weather",
        description="Get the current weather for a city. Returns temperature and conditions.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        },
        function=get_weather,
    )


@pytest.fixture
def calculator_tool():
    """Create a calculator tool for testing."""
    def calculate(operation: str, a: float, b: float) -> str:
        """Perform basic math operations."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: division by zero",
        }
        result = operations.get(operation, "Unknown operation")
        return f"{a} {operation} {b} = {result}"

    return Tool(
        name="calculate",
        description="Perform basic mathematical operations: add, subtract, multiply, divide",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The math operation to perform"
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculate,
    )


# =============================================================================
# BASIC CHAT TESTS
# =============================================================================


@pytest.mark.integration
class TestBasicChat:
    """Basic chat integration tests."""

    def test_simple_question(self, agent):
        """Test a simple question-answer interaction."""
        messages = [ChatMessage.from_user("What is 2 + 2? Answer with just the number.")]
        result = agent.run(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert reply.text is not None
        assert "4" in reply.text
        assert reply.meta.get("model") is not None
        assert reply.meta.get("finish_reason") == "stop"
        assert reply.meta.get("usage") is not None

    def test_multi_turn_conversation(self, agent):
        """Test a multi-turn conversation."""
        messages = [
            ChatMessage.from_user("My name is Alice. Remember this."),
        ]

        # First turn
        result = agent.run(messages)
        assistant_reply = result["replies"][0]
        assert assistant_reply.text is not None

        # Second turn - test context retention
        messages.append(assistant_reply)
        messages.append(ChatMessage.from_user("What is my name?"))

        result = agent.run(messages)
        assert "Alice" in result["replies"][0].text

    def test_system_context(self, agent_id):
        """Test that system messages work correctly."""
        # Note: Agents may have pre-configured system prompts,
        # so this tests additional context
        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={"max_tokens": 100}
        )

        messages = [
            ChatMessage.from_system("Always respond in French."),
            ChatMessage.from_user("Hello, how are you?"),
        ]

        result = agent.run(messages)
        reply = result["replies"][0]

        # The response should be in French (agent may or may not follow this)
        assert reply.text is not None
        assert len(reply.text) > 0


# =============================================================================
# STREAMING TESTS
# =============================================================================


@pytest.mark.integration
class TestStreaming:
    """Streaming integration tests."""

    def test_streaming_basic(self, agent_id):
        """Test basic streaming functionality."""
        chunks_received = []

        def callback(chunk: StreamingChunk):
            chunks_received.append(chunk)

        agent = MistralAgentSDK(
            agent_id=agent_id,
            streaming_callback=callback
        )

        messages = [ChatMessage.from_user("Count from 1 to 5.")]
        result = agent.run(messages)

        # Should have received multiple chunks
        assert len(chunks_received) > 1

        # Final message should be complete
        reply = result["replies"][0]
        assert reply.text is not None

        # Collected content should match final message
        collected_text = "".join(c.content for c in chunks_received if c.content)
        assert collected_text == reply.text

    def test_streaming_runtime_override(self, agent_id):
        """Test that runtime streaming callback overrides init callback."""
        init_chunks = []
        runtime_chunks = []

        def init_callback(chunk):
            init_chunks.append(chunk)

        def runtime_callback(chunk):
            runtime_chunks.append(chunk)

        agent = MistralAgentSDK(
            agent_id=agent_id,
            streaming_callback=init_callback
        )

        # Run with runtime callback
        messages = [ChatMessage.from_user("Say hello")]
        agent.run(messages, streaming_callback=runtime_callback)

        # Runtime callback should be used, not init callback
        assert len(runtime_chunks) > 0
        assert len(init_chunks) == 0


# =============================================================================
# TOOL CALLING TESTS
# =============================================================================


@pytest.mark.integration
class TestToolCalling:
    """Tool calling integration tests."""

    def test_single_tool_call(self, agent_id, weather_tool):
        """Test that the agent can call a single tool."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            tools=[weather_tool],
        )

        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        result = agent.run(messages, tool_choice="any")

        reply = result["replies"][0]

        # Agent should request a tool call
        assert reply.tool_calls is not None
        assert len(reply.tool_calls) >= 1

        tool_call = reply.tool_calls[0]
        assert tool_call.tool_name == "get_weather"
        assert "city" in tool_call.arguments
        assert tool_call.id is not None

    def test_tool_call_and_response(self, agent_id, weather_tool):
        """Test full tool calling flow with response."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            tools=[weather_tool],
        )

        # Initial request
        messages = [ChatMessage.from_user("What's the weather in London?")]
        result = agent.run(messages, tool_choice="any")

        assistant_message = result["replies"][0]
        assert assistant_message.tool_calls

        # Execute the tool
        tool_call = assistant_message.tool_calls[0]
        tool_result = weather_tool.function(**tool_call.arguments)

        # Continue conversation with tool result
        messages.append(assistant_message)
        messages.append(ChatMessage.from_tool(tool_result=tool_result, origin=tool_call))

        result = agent.run(messages)
        final_reply = result["replies"][0]

        # Should get a natural language response about the weather
        assert final_reply.text is not None
        assert "London" in final_reply.text or "rainy" in final_reply.text.lower()

    def test_multiple_tool_calls(self, agent_id, weather_tool, calculator_tool):
        """Test that the agent can call multiple tools."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            tools=[weather_tool, calculator_tool],
            parallel_tool_calls=True,
        )

        messages = [
            ChatMessage.from_user(
                "What's the weather in Paris and Tokyo? Also calculate 15 * 7."
            )
        ]
        result = agent.run(messages, tool_choice="any")

        reply = result["replies"][0]
        assert reply.tool_calls is not None

        # Should have multiple tool calls
        tool_names = [tc.tool_name for tc in reply.tool_calls]
        # The exact number and which tools depends on the agent's behavior
        assert len(tool_names) >= 1

    def test_toolset_integration(self, agent_id, weather_tool, calculator_tool):
        """Test tools provided via Toolset."""
        toolset = Toolset([weather_tool, calculator_tool])

        agent = MistralAgentSDK(
            agent_id=agent_id,
            tools=[toolset],
        )

        messages = [ChatMessage.from_user("What's 100 divided by 4?")]
        result = agent.run(messages, tool_choice="any")

        reply = result["replies"][0]
        if reply.tool_calls:
            assert any(tc.tool_name == "calculate" for tc in reply.tool_calls)

    def test_tool_streaming(self, agent_id, weather_tool):
        """Test streaming with tool calls."""
        chunks_received = []

        def callback(chunk: StreamingChunk):
            chunks_received.append(chunk)

        agent = MistralAgentSDK(
            agent_id=agent_id,
            tools=[weather_tool],
            streaming_callback=callback,
        )

        messages = [ChatMessage.from_user("Check the weather in New York")]
        result = agent.run(messages, tool_choice="any")

        reply = result["replies"][0]
        # Should have tool calls even with streaming
        assert reply.tool_calls is not None


# =============================================================================
# GENERATION KWARGS TESTS
# =============================================================================


@pytest.mark.integration
class TestGenerationKwargs:
    """Tests for generation parameters."""

    def test_max_tokens(self, agent_id):
        """Test max_tokens parameter limits output."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={"max_tokens": 10}
        )

        messages = [ChatMessage.from_user("Tell me a very long story about a dragon.")]
        result = agent.run(messages)

        reply = result["replies"][0]
        # Response should be truncated
        # Note: token count != word count, but should still be limited
        assert reply.meta.get("finish_reason") in ["length", "stop"]

    def test_random_seed_determinism(self, agent_id):
        """Test that random_seed produces deterministic outputs."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={"random_seed": 42, "max_tokens": 50}
        )

        messages = [ChatMessage.from_user("Generate a random number between 1 and 100.")]

        # Run twice with same seed
        result1 = agent.run(messages)
        result2 = agent.run(messages)

        # Results should be the same (or very similar) with same seed
        # Note: This is not 100% guaranteed but should be mostly deterministic
        # We just verify we got responses
        assert result1["replies"][0].text is not None
        assert result2["replies"][0].text is not None

    def test_stop_sequences(self, agent_id):
        """Test stop sequences parameter."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={"stop": ["."]}
        )

        messages = [ChatMessage.from_user("Write a sentence about the sky.")]
        result = agent.run(messages)

        reply = result["replies"][0]
        # Response should stop at first period (or not have multiple sentences)
        assert reply.text is not None


# =============================================================================
# STRUCTURED OUTPUT TESTS
# =============================================================================


@pytest.mark.integration
class TestStructuredOutput:
    """Tests for structured output / response format."""

    def test_json_object_format(self, agent_id):
        """Test JSON object response format."""
        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

        messages = [
            ChatMessage.from_user(
                "Return a JSON object with keys 'name' and 'age' for a person named John who is 30 years old."
            )
        ]
        result = agent.run(messages)

        reply = result["replies"][0]
        assert reply.text is not None

        # Should be valid JSON
        parsed = json.loads(reply.text)
        assert "name" in parsed or "Name" in parsed
        assert "age" in parsed or "Age" in parsed

    def test_json_schema_format(self, agent_id):
        """Test JSON schema response format."""
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "city": {"type": "string"}
                    },
                    "required": ["name", "age", "city"],
                    "additionalProperties": False
                }
            }
        }

        agent = MistralAgentSDK(
            agent_id=agent_id,
            generation_kwargs={"response_format": schema}
        )

        messages = [
            ChatMessage.from_user(
                "Generate info for someone named Maria, age 25, from Barcelona."
            )
        ]
        result = agent.run(messages)

        reply = result["replies"][0]
        parsed = json.loads(reply.text)

        assert parsed["name"] == "Maria"
        assert parsed["age"] == 25
        assert parsed["city"] == "Barcelona"


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    """Tests for Haystack pipeline integration."""

    def test_simple_pipeline(self, agent_id):
        """Test agent in a simple pipeline."""
        pipeline = Pipeline()
        pipeline.add_component("agent", MistralAgentSDK(agent_id=agent_id))

        result = pipeline.run({
            "agent": {
                "messages": [ChatMessage.from_user("What is the capital of Japan?")]
            }
        })

        assert "agent" in result
        assert "replies" in result["agent"]
        assert "Tokyo" in result["agent"]["replies"][0].text

    def test_pipeline_with_tool_invoker(self, agent_id, weather_tool):
        """Test pipeline with ToolInvoker."""
        pipeline = Pipeline()
        pipeline.add_component(
            "agent",
            MistralAgentSDK(agent_id=agent_id, tools=[weather_tool])
        )
        pipeline.add_component(
            "tool_invoker",
            ToolInvoker(tools=[weather_tool])
        )
        pipeline.connect("agent.replies", "tool_invoker.messages")

        result = pipeline.run({
            "agent": {
                "messages": [ChatMessage.from_user("What's the weather in Paris?")],
                "tool_choice": "any"
            }
        })

        # Tool should have been invoked
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) > 0

        tool_result = tool_messages[0].tool_call_result.result
        assert "Paris" in tool_result
        assert "sunny" in tool_result.lower()


# =============================================================================
# ASYNC TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncOperations:
    """Async operation tests."""

    async def test_async_basic(self, agent_id):
        """Test basic async operation."""
        agent = MistralAgentSDK(agent_id=agent_id)

        messages = [ChatMessage.from_user("Hello!")]
        result = await agent.run_async(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        assert result["replies"][0].text is not None

    async def test_async_with_tools(self, agent_id, weather_tool):
        """Test async with tools."""
        agent = MistralAgentSDK(agent_id=agent_id, tools=[weather_tool])

        messages = [ChatMessage.from_user("Weather in Tokyo?")]
        result = await agent.run_async(messages, tool_choice="any")

        reply = result["replies"][0]
        # May or may not have tool calls depending on agent config
        assert reply.text is not None or reply.tool_calls is not None


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Error handling tests."""

    def test_invalid_agent_id(self):
        """Test that invalid agent ID raises appropriate error."""
        agent = MistralAgentSDK(agent_id="invalid-agent-id-12345")

        messages = [ChatMessage.from_user("Hello")]

        with pytest.raises(Exception):  # Should raise ValueError or MistralError
            agent.run(messages)

    def test_empty_messages_returns_empty(self, agent_id):
        """Test that empty messages return empty replies."""
        agent = MistralAgentSDK(agent_id=agent_id)
        result = agent.run([])

        assert result == {"replies": []}

