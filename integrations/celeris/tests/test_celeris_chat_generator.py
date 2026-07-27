import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ImageContent, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.celeris.chat.chat_generator import (
    CELERIS_CONTEXT_LIMIT,
    CELERIS_DEFAULT_MAX_TOKENS,
    CelerisChatGenerator,
)


class CollectorCallback:
    """
    Callback to collect streaming chunks for testing purposes.
    """

    def __init__(self):
        self.chunks = []

    def __call__(self, chunk: StreamingChunk) -> None:
        self.chunks.append(chunk)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather,
    )

    return [tool]


@pytest.fixture
def mock_chat_completion():
    """
    Mock the Celeris API completion response and reuse it for tests.
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="celeris-1",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestCelerisChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "test-api-key")
        component = CelerisChatGenerator()
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "celeris-1"
        assert component.api_base_url == "https://inference.celeris.ai/celeris-1/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "test-api-key")
        component = CelerisChatGenerator()
        component.warm_up()  # with haystack-ai >= 3.0 the client is created during warm-up
        assert component.client.api_key == "test-api-key"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("CELERIS_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            # haystack-ai 2.x raises at init; haystack-ai >= 3.0 raises when the client is created in warm_up
            component = CelerisChatGenerator()
            component.warm_up()

    def test_init_with_parameters(self):
        component = CelerisChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="celeris-1",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 512, "temperature": 0.5},
            timeout=10,
            max_retries=2,
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "celeris-1"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 512, "temperature": 0.5}
        assert component.timeout == 10
        assert component.max_retries == 2

    def test_init_warns_when_model_changed_but_base_url_is_not(self, monkeypatch, caplog):
        # Celeris puts the model in the endpoint path, so the two must be changed together
        monkeypatch.setenv("CELERIS_API_KEY", "test-api-key")
        CelerisChatGenerator(model="some-other-model")
        assert "api_base_url" in caplog.text

    def test_init_does_not_warn_when_base_url_changed_too(self, monkeypatch, caplog):
        monkeypatch.setenv("CELERIS_API_KEY", "test-api-key")
        CelerisChatGenerator(model="some-other-model", api_base_url="https://inference.celeris.ai/some-other-model/v1")
        assert "api_base_url" not in caplog.text

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "test-api-key")
        component = CelerisChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.celeris.chat.chat_generator.CelerisChatGenerator"
        )

        assert data["init_parameters"] == {
            "api_key": {"env_vars": ["CELERIS_API_KEY"], "strict": True, "type": "env_var"},
            "model": "celeris-1",
            "streaming_callback": None,
            "api_base_url": "https://inference.celeris.ai/celeris-1/v1",
            "generation_kwargs": {},
            "tools": None,
            "tools_strict": False,
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = CelerisChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="celeris-1",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 512, "temperature": 0.5},
            timeout=10,
            max_retries=10,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "celeris-1",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 512, "temperature": 0.5},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "tools_strict": False,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.celeris.chat.chat_generator.CelerisChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["CELERIS_API_KEY"], "strict": True, "type": "env_var"},
                "model": "celeris-1",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 512, "temperature": 0.5},
                "tools": None,
                "tools_strict": False,
                "timeout": 10,
                "max_retries": 10,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = CelerisChatGenerator.from_dict(data)
        assert component.model == "celeris-1"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 512, "temperature": 0.5}
        assert component.api_key == Secret.from_env_var("CELERIS_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_serde_in_pipeline(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "test-key")

        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        generator = CelerisChatGenerator(
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()

        # the Tool serialization format is owned by haystack-ai and varies across its versions; the
        # dumps/loads round-trip below covers the tools, so exclude them from the pinned-dict comparison
        tools_entries = pipeline_dict["components"]["generator"]["init_parameters"].pop("tools")
        assert len(tools_entries) == 1
        assert pipeline_dict["components"]["generator"] == {
            "type": "haystack_integrations.components.generators.celeris.chat.chat_generator.CelerisChatGenerator",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["CELERIS_API_KEY"], "strict": True},
                "model": "celeris-1",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "api_base_url": "https://inference.celeris.ai/celeris-1/v1",
                "generation_kwargs": {"temperature": 0.7},
                "tools_strict": False,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

        loaded_generator = new_pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.api_base_url == generator.api_base_url
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback == generator.streaming_callback
        assert len(loaded_generator.tools) == len(generator.tools)
        assert loaded_generator.tools[0].name == generator.tools[0].name

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        response = component.run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["model"] == "celeris-1"

        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 512, "temperature": 0.5})
        component.run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.5


class TestMaxTokensQuantization:
    """`max_tokens` must be 1 (warm ping) or a positive multiple of 256, otherwise Celeris returns a 400."""

    def test_default_max_tokens_is_sent(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        CelerisChatGenerator().run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == CELERIS_DEFAULT_MAX_TOKENS
        assert CELERIS_DEFAULT_MAX_TOKENS % 256 == 0

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(1, 1), (2, 256), (100, 256), (256, 256), (257, 512), (512, 512), (1000, 1024), (7000, 7168)],
    )
    def test_rounds_up_to_multiple_of_256(self, requested, expected):
        assert CelerisChatGenerator._resolve_max_tokens(requested_max_tokens=requested, prompt_tokens=0) == expected

    def test_unquantized_max_tokens_is_rounded_up_at_run_time(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        CelerisChatGenerator(generation_kwargs={"max_tokens": 100}).run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 256

    def test_warm_ping_max_tokens_is_sent_unchanged(self, chat_messages, mock_chat_completion, monkeypatch):
        # 1 is the only non-multiple-of-256 value Celeris accepts, so quantization must leave it alone
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        CelerisChatGenerator(generation_kwargs={"max_tokens": 1}).run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 1

    def test_quantization_applied_to_run_generation_kwargs(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 100})
        component.run(chat_messages, generation_kwargs={"max_tokens": 300})

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 512

    def test_generation_kwargs_are_not_mutated(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 100})
        component.run(chat_messages)
        assert component.generation_kwargs == {"max_tokens": 100}

    @pytest.mark.parametrize("requested", [0, -256])
    def test_rejects_non_positive_max_tokens(self, requested):
        with pytest.raises(ValueError, match="positive integer"):
            CelerisChatGenerator._resolve_max_tokens(requested_max_tokens=requested, prompt_tokens=0)

    def test_rejects_non_int_max_tokens(self):
        with pytest.raises(ValueError, match="must be an int"):
            CelerisChatGenerator._resolve_max_tokens(requested_max_tokens=100.5, prompt_tokens=0)


class TestSharedContextBudget:
    """Prompt and completion share a single 8192-token budget."""

    def test_caps_max_tokens_to_remaining_budget(self):
        # 7800 prompt tokens leaves 392 tokens, which floors to a single 256-token block
        resolved = CelerisChatGenerator._resolve_max_tokens(requested_max_tokens=4096, prompt_tokens=7800)
        assert resolved == 256
        assert 7800 + resolved <= CELERIS_CONTEXT_LIMIT

    def test_cap_keeps_prompt_plus_max_tokens_within_budget(self):
        for prompt_tokens in range(0, CELERIS_CONTEXT_LIMIT - 256, 337):
            resolved = CelerisChatGenerator._resolve_max_tokens(
                requested_max_tokens=CELERIS_CONTEXT_LIMIT, prompt_tokens=prompt_tokens
            )
            assert resolved % 256 == 0
            assert prompt_tokens + resolved <= CELERIS_CONTEXT_LIMIT

    def test_raises_when_prompt_leaves_no_room(self):
        with pytest.raises(ValueError, match="fewer than 256 tokens"):
            CelerisChatGenerator._resolve_max_tokens(requested_max_tokens=256, prompt_tokens=8100)

    def test_long_prompt_shrinks_max_tokens_at_run_time(self, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        long_prompt = "a" * 23_400  # roughly 7.8K tokens with the conservative estimator
        component.run([ChatMessage.from_user(long_prompt)])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 256

    def test_oversized_prompt_raises_at_run_time(self, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        with pytest.raises(ValueError, match="fewer than 256 tokens"):
            component.run([ChatMessage.from_user("a" * 24_600)])
        mock_chat_completion.assert_not_called()

    def test_tool_definitions_count_towards_the_prompt(self, tools):
        messages = [ChatMessage.from_user("hello")]
        without_tools = CelerisChatGenerator._estimate_prompt_tokens(messages=messages, tools=[])
        with_tools = CelerisChatGenerator._estimate_prompt_tokens(messages=messages, tools=tools)
        assert with_tools > without_tools

    def test_tool_calls_and_results_count_towards_the_prompt(self):
        # a tool-calling turn carries no plain text, so it must still be measured via its calls and results
        tool_call = ToolCall(id="1", tool_name="weather", arguments={"city": "Paris"})
        baseline = CelerisChatGenerator._estimate_prompt_tokens(messages=[ChatMessage.from_assistant("")], tools=[])

        with_call = CelerisChatGenerator._estimate_prompt_tokens(
            messages=[ChatMessage.from_assistant(tool_calls=[tool_call])], tools=[]
        )
        with_result = CelerisChatGenerator._estimate_prompt_tokens(
            messages=[ChatMessage.from_tool(tool_result="sunny and 32C", origin=tool_call)], tools=[]
        )
        assert with_call > baseline
        assert with_result > baseline

    def test_message_name_counts_towards_the_prompt(self):
        without_name = CelerisChatGenerator._estimate_prompt_tokens(messages=[ChatMessage.from_user("hello")], tools=[])
        with_name = CelerisChatGenerator._estimate_prompt_tokens(
            messages=[ChatMessage.from_user("hello", name="a-very-long-participant-name")], tools=[]
        )
        assert with_name > without_name


class TestUnsupportedFeatures:
    @pytest.mark.parametrize(
        "response_format",
        [{"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}],
    )
    def test_rejects_response_format(self, chat_messages, mock_chat_completion, monkeypatch, response_format):
        # covers both JSON mode and JSON-schema structured output
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        with pytest.raises(ValueError, match="does not support the 'response_format' parameter") as exc_info:
            component.run(chat_messages, generation_kwargs={"response_format": response_format})

        # the rejection must point users at the supported alternative
        assert "tool calling" in str(exc_info.value)
        mock_chat_completion.assert_not_called()

    def test_rejects_response_format_set_at_init(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"response_format": {"type": "json_object"}})
        with pytest.raises(ValueError, match="does not support the 'response_format' parameter"):
            component.run(chat_messages)
        mock_chat_completion.assert_not_called()

    def test_forwards_logprobs(self, chat_messages, mock_chat_completion, monkeypatch):
        # The Celeris API reference lists `logprobs` as unsupported, but the endpoint accepts it and
        # returns per-token values, so the component must not reject it client-side.
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        component.run(chat_messages, generation_kwargs={"logprobs": True, "top_logprobs": 3})

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 3

    def test_rejects_tool_choice_required(self, chat_messages, tools, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(tools=tools)
        with pytest.raises(ValueError, match="does not support tool_choice='required'"):
            component.run(chat_messages, generation_kwargs={"tool_choice": "required"})
        mock_chat_completion.assert_not_called()

    def test_rejects_tool_choice_required_set_at_init(self, chat_messages, tools, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(tools=tools, generation_kwargs={"tool_choice": "required"})
        with pytest.raises(ValueError, match="does not support tool_choice='required'"):
            component.run(chat_messages)
        mock_chat_completion.assert_not_called()

    @pytest.mark.parametrize("tool_choice", ["auto", "none", {"type": "function", "function": {"name": "weather"}}])
    def test_allows_supported_tool_choice(self, chat_messages, tools, mock_chat_completion, monkeypatch, tool_choice):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(tools=tools)
        component.run(chat_messages, generation_kwargs={"tool_choice": tool_choice})

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["tool_choice"] == tool_choice

    def test_rejects_image_input(self, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        image_message = ChatMessage.from_user(
            content_parts=["What is in this image?", ImageContent(base64_image="aGk=", mime_type="image/png")]
        )
        with pytest.raises(ValueError, match="text-only"):
            component.run([image_message])
        mock_chat_completion.assert_not_called()

    def test_tools_are_forwarded(self, tools, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(tools=tools)
        args = component._prepare_api_call(messages=[ChatMessage.from_user("hi")])

        assert args["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "useful to determine the weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    def test_streaming_multiple_responses_raises(self, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        with pytest.raises(ValueError, match="Cannot stream multiple responses"):
            component._prepare_api_call(
                messages=[ChatMessage.from_user("hi")],
                streaming_callback=print_streaming_chunk,
                generation_kwargs={"n": 2},
            )


@pytest.mark.skipif(
    not os.environ.get("CELERIS_API_KEY", None),
    reason="Export an env var called CELERIS_API_KEY containing the Celeris API key to run this test.",
)
@pytest.mark.integration
class TestCelerisChatGeneratorLiveRun:
    def test_live_run(self):
        component = CelerisChatGenerator()
        results = component.run([ChatMessage.from_user("What's the capital of France? Answer with one word.")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text
        assert "Paris" in message.text
        assert message.meta["model"] == "celeris-1"

    def test_live_run_with_unquantized_max_tokens(self):
        # 100 is not a multiple of 256; the component must round it up before the request is sent
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 100})
        results = component.run([ChatMessage.from_user("What's the capital of France? Answer with one word.")])

        assert len(results["replies"]) == 1
        assert results["replies"][0].text

    def test_live_run_streaming(self):
        callback = CollectorCallback()
        component = CelerisChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France? Answer with one word.")])

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0].text
        assert len(callback.chunks) > 0

    def test_live_run_with_tools(self, tools):
        component = CelerisChatGenerator(tools=tools)
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        assert message.tool_call.tool_name == "weather"
        assert message.tool_call.arguments == {"city": "Paris"}
