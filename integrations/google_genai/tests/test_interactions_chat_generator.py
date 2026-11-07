# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import interactions as gi
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ImageContent, StreamingChunk
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from pydantic import BaseModel

from haystack_integrations.components.generators.google_genai import GoogleGenAIInteractionsChatGenerator


def weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: 22°C, sunny"


def streaming_callback(chunk: StreamingChunk) -> None:
    pass


@pytest.fixture
def tools():
    return [
        Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
    ]


@pytest.fixture
def interaction():
    return gi.Interaction.model_validate(
        {
            "id": "int_1",
            "status": "completed",
            "model": "gemini-3.8-flash",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Paris."}]}],
            "usage": {"total_input_tokens": 5, "total_output_tokens": 2, "total_tokens": 7},
        }
    )


@pytest.fixture
def stream_events():
    interaction = {"id": "int_1"}
    return [
        gi.InteractionCreatedEvent.model_validate(
            {"event_type": "interaction.created", "interaction": {**interaction, "status": "in_progress"}}
        ),
        gi.StepStart.model_validate({"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}),
        gi.StepDelta.model_validate(
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Paris."}}
        ),
        gi.InteractionCompletedEvent.model_validate(
            {"event_type": "interaction.completed", "interaction": {**interaction, "status": "completed"}}
        ),
    ]


async def _async_iter(items):
    for item in items:
        yield item


def test_supported_models_is_non_empty_list_of_strings():
    models = GoogleGenAIInteractionsChatGenerator.SUPPORTED_MODELS
    assert models
    assert all(isinstance(model, str) for model in models)


class TestInitSerDe:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIInteractionsChatGenerator()
        assert component._model == "gemini-3.8-flash"
        assert component._api == "gemini"
        assert component._generation_kwargs == {}
        assert component._streaming_callback is None
        assert component._tools is None
        assert component._api_key.resolve_value() == "test-api-key"
        assert component._client is None
        assert component._async_client is None

    def test_init_fail_with_duplicate_tool_names(self, tools):
        with pytest.raises(ValueError, match="Duplicate tool names"):
            GoogleGenAIInteractionsChatGenerator(tools=[tools[0], tools[0]])

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        assert GoogleGenAIInteractionsChatGenerator().to_dict() == {
            "type": "haystack_integrations.components.generators.google_genai.chat.interactions_chat_generator."
            "GoogleGenAIInteractionsChatGenerator",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
                "model": "gemini-3.8-flash",
                "generation_kwargs": {},
                "streaming_callback": None,
                "tools": None,
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_serde_round_trip_with_non_default_params(self, monkeypatch, tools):
        monkeypatch.setenv("MY_GOOGLE_KEY", "custom-key")
        component = GoogleGenAIInteractionsChatGenerator(
            api_key=Secret.from_env_var("MY_GOOGLE_KEY"),
            api="vertex",
            vertex_ai_project="my-project",
            vertex_ai_location="europe-west1",
            model="gemini-2.5-flash",
            generation_kwargs={"generation_config": {"temperature": 0.2}, "store": False},
            streaming_callback=streaming_callback,
            tools=Toolset(tools),
            timeout=10.0,
            max_retries=2,
        )
        restored = GoogleGenAIInteractionsChatGenerator.from_dict(component.to_dict())
        assert restored._api_key.resolve_value() == "custom-key"
        assert restored._api == "vertex"
        assert restored._vertex_ai_project == "my-project"
        assert restored._vertex_ai_location == "europe-west1"
        assert restored._model == "gemini-2.5-flash"
        assert restored._generation_kwargs == {"generation_config": {"temperature": 0.2}, "store": False}
        assert restored._streaming_callback is streaming_callback
        assert isinstance(restored._tools, Toolset)
        assert [tool.name for tool in restored._tools] == ["weather"]
        assert restored._timeout == 10.0
        assert restored._max_retries == 2

    def test_to_dict_with_pydantic_response_format(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        class City(BaseModel):
            name: str

        component = GoogleGenAIInteractionsChatGenerator(generation_kwargs={"response_format": City})
        data = component.to_dict()
        assert data["init_parameters"]["generation_kwargs"]["response_format"] == City.model_json_schema()
        assert GoogleGenAIInteractionsChatGenerator.from_dict(data)._generation_kwargs == {
            "response_format": City.model_json_schema()
        }


class TestComponentLifecycle:
    @patch("haystack_integrations.components.generators.google_genai.chat.interactions_chat_generator._get_client")
    def test_sync_lifecycle(self, mock_get_client):
        component = GoogleGenAIInteractionsChatGenerator(api_key=Secret.from_token("test-api-key"))
        client = mock_get_client.return_value

        component.warm_up()
        component.warm_up()
        assert component._client is client
        assert mock_get_client.call_count == 1

        component.close()
        client.close.assert_called_once_with()
        assert component._client is None

    @patch("haystack_integrations.components.generators.google_genai.chat.interactions_chat_generator._get_client")
    async def test_async_lifecycle(self, mock_get_client):
        component = GoogleGenAIInteractionsChatGenerator(api_key=Secret.from_token("test-api-key"))
        client = mock_get_client.return_value
        client.aclose = AsyncMock()

        await component.warm_up_async()
        await component.warm_up_async()
        assert component._async_client is client
        assert mock_get_client.call_count == 1
        assert mock_get_client.call_args.kwargs["async_client"] is True

        await component.close_async()
        client.aclose.assert_awaited_once_with()
        assert component._async_client is None


class TestRun:
    def _component(self, **kwargs):
        component = GoogleGenAIInteractionsChatGenerator(api_key=Secret.from_token("test-api-key"), **kwargs)
        component._client = MagicMock()
        return component

    def test_run(self, interaction):
        component = self._component()
        component._client.interactions.create.return_value = interaction

        result = component.run([ChatMessage.from_system("Be brief."), ChatMessage.from_user("Capital of France?")])

        assert result["replies"][0].text == "Paris."
        assert result["replies"][0].meta["finish_reason"] == "stop"
        # `store` is left to the API default unless the user sets it
        component._client.interactions.create.assert_called_once_with(
            model="gemini-3.8-flash",
            system_instruction="Be brief.",
            input=[{"type": "user_input", "content": [{"type": "text", "text": "Capital of France?"}]}],
        )

    def test_run_merges_generation_kwargs(self, interaction):
        component = self._component(
            generation_kwargs={"generation_config": {"temperature": 0.2, "max_output_tokens": 50}, "store": False}
        )
        component._client.interactions.create.return_value = interaction

        component.run("Hi", generation_kwargs={"generation_config": {"temperature": 0.9}, "service_tier": "flex"})

        kwargs = component._client.interactions.create.call_args.kwargs
        assert kwargs["generation_config"] == {"temperature": 0.9, "max_output_tokens": 50}
        assert kwargs["store"] is False
        assert kwargs["service_tier"] == "flex"

    def test_run_tools(self, interaction, tools):
        init_tool = Tool(name="init_tool", description="Init tool", parameters={"type": "object"}, function=print)
        component = self._component(tools=[init_tool], generation_kwargs={"tools": [{"type": "google_search"}]})
        component._client.interactions.create.return_value = interaction

        component.run("Hi")
        assert [t.get("name") for t in component._client.interactions.create.call_args.kwargs["tools"]] == [
            None,
            "init_tool",
        ]

        # Runtime tools replace the init tools; tools from generation_kwargs are kept
        component.run("Hi", tools=tools)
        assert component._client.interactions.create.call_args.kwargs["tools"] == [
            {"type": "google_search"},
            {
                "type": "function",
                "name": "weather",
                "description": "useful to determine the weather in a given location",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        ]

        # An empty list disables the init tools for this call
        component.run("Hi", tools=[])
        assert component._client.interactions.create.call_args.kwargs["tools"] == [{"type": "google_search"}]

    def test_run_streaming(self, stream_events):
        chunks = []
        component = self._component(streaming_callback=chunks.append)
        component._client.interactions.create.return_value = iter(stream_events)

        result = component.run("Capital of France?")

        assert component._client.interactions.create.call_args.kwargs["stream"] is True
        assert [chunk.content for chunk in chunks] == ["", "", "Paris.", ""]
        assert all(chunk.component_info.type.endswith("GoogleGenAIInteractionsChatGenerator") for chunk in chunks)
        assert result["replies"][0].text == "Paris."
        assert result["replies"][0].meta["interaction_id"] == "int_1"

    def test_run_wraps_api_errors(self):
        component = self._component()
        component._client.interactions.create.side_effect = Exception("quota exceeded")

        with pytest.raises(RuntimeError, match="quota exceeded"):
            component.run("Hi")

    async def test_run_async(self, interaction):
        component = GoogleGenAIInteractionsChatGenerator(api_key=Secret.from_token("test-api-key"))
        component._async_client = MagicMock()
        component._async_client.interactions.create = AsyncMock(return_value=interaction)

        result = await component.run_async("Capital of France?", generation_kwargs={"store": False})

        assert result["replies"][0].text == "Paris."
        component._async_client.interactions.create.assert_awaited_once_with(
            store=False,
            model="gemini-3.8-flash",
            input=[{"type": "user_input", "content": [{"type": "text", "text": "Capital of France?"}]}],
        )

    async def test_run_async_streaming_with_async_callback(self, stream_events):
        chunks = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = GoogleGenAIInteractionsChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=callback
        )
        component._async_client = MagicMock()
        component._async_client.interactions.create = AsyncMock(return_value=_async_iter(stream_events))

        result = await component.run_async("Capital of France?")

        assert component._async_client.interactions.create.call_args.kwargs["stream"] is True
        assert [chunk.content for chunk in chunks] == ["", "", "Paris.", ""]
        assert result["replies"][0].text == "Paris."

    async def test_run_async_wraps_api_errors(self):
        component = GoogleGenAIInteractionsChatGenerator(api_key=Secret.from_token("test-api-key"))
        component._async_client = MagicMock()
        component._async_client.interactions.create = AsyncMock(side_effect=Exception("quota exceeded"))

        with pytest.raises(RuntimeError, match="quota exceeded"):
            await component.run_async("Hi")


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
class TestInference:
    def test_live_run(self):
        component = GoogleGenAIInteractionsChatGenerator()
        result = component.run([ChatMessage.from_user("What's the capital of France?")])

        reply = result["replies"][0]
        assert "paris" in reply.text.lower()
        assert reply.meta["finish_reason"] == "stop"
        assert reply.meta["interaction_id"]
        assert reply.meta["usage"]["total_tokens"] > 0

    def test_live_run_streaming(self):
        chunks: list[StreamingChunk] = []
        component = GoogleGenAIInteractionsChatGenerator(streaming_callback=chunks.append)
        result = component.run([ChatMessage.from_user("What's the capital of France?")])

        reply = result["replies"][0]
        assert "paris" in reply.text.lower()
        assert "".join(chunk.content for chunk in chunks) == reply.text
        assert reply.meta["finish_reason"] == "stop"

    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_with_thinking_and_tools_multi_turn(self, tools, streaming):
        """Thought signatures and tool calls must round-trip through the stateless history."""
        component = GoogleGenAIInteractionsChatGenerator(
            tools=tools,
            generation_kwargs={"generation_config": {"thinking_level": "low", "thinking_summaries": "auto"}},
            streaming_callback=(lambda chunk: None) if streaming else None,
        )
        messages = [ChatMessage.from_user("What's the weather in Paris and in Rome?")]
        first = component.run(messages)["replies"][0]

        assert {tool_call.arguments["city"] for tool_call in first.tool_calls} == {"Paris", "Rome"}
        assert first.meta["finish_reason"] == "tool_calls"
        assert first.reasoning is not None
        assert first.reasoning.extra["thought_steps"]

        messages.append(first)
        messages.extend(
            ChatMessage.from_tool(tool_result=weather(**tool_call.arguments), origin=tool_call)
            for tool_call in first.tool_calls
        )
        second = component.run(messages)["replies"][0]

        assert "22" in second.text
        assert second.meta["finish_reason"] == "stop"

    def test_live_run_with_structured_output(self):
        class City(BaseModel):
            name: str
            country: str

        component = GoogleGenAIInteractionsChatGenerator(generation_kwargs={"response_format": City})
        result = component.run([ChatMessage.from_user("Tell me about Paris")])

        city = City.model_validate(json.loads(result["replies"][0].text))
        assert city.name == "Paris"
        assert city.country == "France"

    def test_live_run_with_image(self, test_files_path):
        image = ImageContent.from_file_path(test_files_path / "apple.jpg")
        component = GoogleGenAIInteractionsChatGenerator()
        result = component.run(
            [ChatMessage.from_user(content_parts=["What fruit is this? Answer in one word.", image])]
        )

        assert "apple" in result["replies"][0].text.lower()

    def test_live_agent_with_tool(self, tools):
        agent = Agent(chat_generator=GoogleGenAIInteractionsChatGenerator(), tools=tools)
        result = agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")])

        assert "22" in result["last_message"].text


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
class TestAsyncInference:
    async def test_live_run_async(self):
        component = GoogleGenAIInteractionsChatGenerator()
        result = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        reply = result["replies"][0]
        assert "paris" in reply.text.lower()
        assert reply.meta["finish_reason"] == "stop"
        await component.close_async()

    async def test_live_run_async_streaming(self):
        chunks: list[StreamingChunk] = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = GoogleGenAIInteractionsChatGenerator(streaming_callback=callback)
        result = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        reply = result["replies"][0]
        assert "paris" in reply.text.lower()
        assert "".join(chunk.content for chunk in chunks) == reply.text
        await component.close_async()

    async def test_live_run_async_with_thinking_and_tools_multi_turn(self, tools):
        component = GoogleGenAIInteractionsChatGenerator(
            tools=tools,
            generation_kwargs={"generation_config": {"thinking_level": "low", "thinking_summaries": "auto"}},
        )
        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        first = (await component.run_async(messages))["replies"][0]
        assert first.tool_calls

        messages.append(first)
        messages.extend(
            ChatMessage.from_tool(tool_result=weather(**tool_call.arguments), origin=tool_call)
            for tool_call in first.tool_calls
        )
        second = (await component.run_async(messages))["replies"][0]

        assert "22" in second.text
        await component.close_async()
