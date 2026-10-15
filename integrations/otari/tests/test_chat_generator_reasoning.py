# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ReasoningContent, StreamingChunk, ToolCall
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from haystack_integrations.components.generators.otari import OtariChatGenerator

from .conftest import OTARI_API_BASE_URL, requires_api_key

SIGNATURE = {"anthropic": {"signature": "c2lnbmF0dXJl"}}
# The test gateway in docker-compose.yml routes to Cohere, whose reasoning model returns its thinking
COHERE_REASONING_MODEL = "cohere:command-a-reasoning-08-2025"
ANTHROPIC_REASONING_MODEL = "anthropic:claude-sonnet-4-6"
# with a low effort, Claude often answers simple prompts without reasoning
ANTHROPIC_GENERATION_KWARGS = {"reasoning_effort": "high", "max_tokens": 8000}


def _completion(message: ChatCompletionMessage) -> ChatCompletion:
    return ChatCompletion(
        id="foo",
        model="claude-sonnet-4-6",
        object="chat.completion",
        created=1750162525,
        choices=[Choice(finish_reason="stop", index=0, logprobs=None, message=message)],
    )


def _chunk(finish_reason: str | None = None, **delta: object) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="foo",
        model="claude-sonnet-4-6",
        object="chat.completion.chunk",
        created=1750162525,
        choices=[ChunkChoice(delta=ChoiceDelta(**delta), index=0, finish_reason=finish_reason)],
    )


# Otari streams the reasoning first, then the thinking signature in a delta of its own, then the text
STREAM = [
    _chunk(role="assistant"),
    _chunk(reasoning="Paris is the "),
    _chunk(reasoning="capital."),
    _chunk(extra_content=SIGNATURE),
    _chunk(content="Paris"),
    _chunk(content=" it is.", finish_reason="stop"),
]


class AsyncStreamStub:
    """An async iterator over chunks, with the close() that the async stream handler calls on cancellation."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self):
        pass


@pytest.fixture
def generator():
    return OtariChatGenerator(api_key=Secret.from_token("test-api-key"), model="anthropic:claude-sonnet-4-6")


class TestReasoning:
    def test_run_reads_reasoning_and_signature(self, generator):
        message = ChatCompletionMessage(
            role="assistant", content="Paris.", reasoning="Paris is the capital.", extra_content=SIGNATURE
        )
        with patch("openai.resources.chat.completions.Completions.create", return_value=_completion(message)):
            reply = generator.run([ChatMessage.from_user("What's the capital of France?")])["replies"][0]

        assert reply.text == "Paris."
        assert reply.reasoning == ReasoningContent(
            reasoning_text="Paris is the capital.", extra={"extra_content": SIGNATURE}
        )
        assert reply.meta["finish_reason"] == "stop"

    def test_run_without_reasoning_is_unchanged(self, generator):
        message = ChatCompletionMessage(role="assistant", content="Paris.")
        with patch("openai.resources.chat.completions.Completions.create", return_value=_completion(message)):
            reply = generator.run([ChatMessage.from_user("What's the capital of France?")])["replies"][0]

        assert reply.text == "Paris."
        assert reply.reasoning is None

    def test_run_streaming_reads_reasoning_and_signature(self, generator):
        chunks: list[StreamingChunk] = []
        with patch("openai.resources.chat.completions.Completions.create", return_value=iter(STREAM)):
            reply = generator.run(
                [ChatMessage.from_user("What's the capital of France?")], streaming_callback=chunks.append
            )["replies"][0]

        reasoning_chunks = [chunk for chunk in chunks if chunk.reasoning]
        assert [chunk.reasoning.reasoning_text for chunk in reasoning_chunks] == ["Paris is the ", "capital.", ""]
        # print_streaming_chunk prints a header on the first reasoning chunk and on the first text chunk
        assert [chunk.start for chunk in reasoning_chunks] == [True, False, False]
        assert [chunk.start for chunk in chunks if chunk.content] == [True, False]

        assert reply.text == "Paris it is."
        assert reply.reasoning == ReasoningContent(
            reasoning_text="Paris is the capital.", extra={"extra_content": SIGNATURE}
        )

    @pytest.mark.asyncio
    async def test_run_async_streaming_reads_reasoning_and_signature(self, generator):
        chunks: list[StreamingChunk] = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            new_callable=AsyncMock,
            return_value=AsyncStreamStub(STREAM),
        ):
            result = await generator.run_async(
                [ChatMessage.from_user("What's the capital of France?")], streaming_callback=callback
            )

        reply = result["replies"][0]
        assert [chunk.reasoning.reasoning_text for chunk in chunks if chunk.reasoning] == [
            "Paris is the ",
            "capital.",
            "",
        ]
        assert reply.text == "Paris it is."
        assert reply.reasoning == ReasoningContent(
            reasoning_text="Paris is the capital.", extra={"extra_content": SIGNATURE}
        )

    def test_run_sends_reasoning_back(self, generator):
        tool_call = ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})
        messages = [
            ChatMessage.from_user("What's the weather in Paris?"),
            ChatMessage.from_assistant(
                tool_calls=[tool_call],
                reasoning=ReasoningContent(
                    reasoning_text="I need the weather tool.", extra={"extra_content": SIGNATURE}
                ),
            ),
            ChatMessage.from_tool(tool_result="22° C and sunny", origin=tool_call),
            ChatMessage.from_assistant("It's sunny."),
            ChatMessage.from_user("And tomorrow?"),
        ]
        reply = ChatCompletionMessage(role="assistant", content="Probably sunny too.")
        with patch(
            "openai.resources.chat.completions.Completions.create", return_value=_completion(reply)
        ) as mock_create:
            generator.run(messages)

        sent = mock_create.call_args.kwargs["messages"]
        assert sent[1]["reasoning"] == "I need the weather tool."
        assert sent[1]["extra_content"] == SIGNATURE
        assert sent[1]["tool_calls"][0]["id"] == "call_1"
        # messages without reasoning are sent as before
        assert "reasoning" not in sent[3]
        assert sent[0] == messages[0].to_openai_dict_format()

    def test_run_does_not_send_back_reasoning_without_provider_data(self, generator):
        # some providers, such as Cohere, reject reasoning in a request
        messages = [
            ChatMessage.from_user("What is 17 * 23?"),
            ChatMessage.from_assistant("391", reasoning=ReasoningContent(reasoning_text="17 * 23 = 391")),
            ChatMessage.from_user("Now add 9."),
        ]
        reply = ChatCompletionMessage(role="assistant", content="400")
        with patch(
            "openai.resources.chat.completions.Completions.create", return_value=_completion(reply)
        ) as mock_create:
            generator.run(messages)

        sent = mock_create.call_args.kwargs["messages"]
        assert sent[1] == messages[1].to_openai_dict_format()
        assert "reasoning" not in sent[1]


# "cohere" and "anthropic" in the test names let the CI step against otari.ai skip them, since that organization has
# no Cohere or Anthropic key
class TestReasoningLive:
    @requires_api_key
    @pytest.mark.integration
    def test_live_run_with_cohere_reasoning(self):
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=COHERE_REASONING_MODEL,
            generation_kwargs={"reasoning_effort": "low"},
        )
        reply = generator.run([ChatMessage.from_user("What is 17 * 23? Answer with the number only.")])["replies"][0]

        assert "391" in reply.text
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_streaming_with_cohere_reasoning(self):
        chunks: list[StreamingChunk] = []
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=COHERE_REASONING_MODEL,
            streaming_callback=chunks.append,
            generation_kwargs={"reasoning_effort": "low"},
        )
        reply = generator.run([ChatMessage.from_user("What is 17 * 23? Answer with the number only.")])["replies"][0]

        assert "391" in reply.text
        assert any(chunk.reasoning for chunk in chunks)
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text

    @requires_api_key
    @pytest.mark.integration
    def test_live_multi_turn_with_cohere_reasoning(self):
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=COHERE_REASONING_MODEL,
            generation_kwargs={"reasoning_effort": "low"},
        )
        messages = [ChatMessage.from_user("What is 17 * 23? Answer with the number only.")]
        first = generator.run(messages)["replies"][0]
        assert first.reasoning is not None

        # Cohere's reasoning comes without provider data to replay, so it isn't sent back, which Cohere would reject
        assert "extra_content" not in first.reasoning.extra
        second = generator.run([*messages, first, ChatMessage.from_user("Now add 9. Answer with the number only.")])
        assert "400" in second["replies"][0].text

    @requires_api_key
    @pytest.mark.integration
    def test_live_tool_loop_with_cohere_reasoning(self, tools):
        # Cohere returns no reasoning with a tool call through Otari, so this checks the tool loop of a reasoning model
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=COHERE_REASONING_MODEL,
            tools=tools,
            generation_kwargs={"reasoning_effort": "low"},
        )
        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        tool_message = generator.run(messages)["replies"][0]
        assert tool_message.tool_call is not None

        final = generator.run(
            [
                *messages,
                tool_message,
                ChatMessage.from_tool(tool_result="22° C and sunny", origin=tool_message.tool_call),
            ]
        )["replies"][0]
        assert "22" in final.text

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_streaming_with_anthropic_reasoning(self):
        chunks: list[StreamingChunk] = []
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=ANTHROPIC_REASONING_MODEL,
            streaming_callback=chunks.append,
            generation_kwargs=ANTHROPIC_GENERATION_KWARGS,
        )
        reply = generator.run([ChatMessage.from_user("What is 17 * 23? Answer with the number only.")])["replies"][0]

        assert "391" in reply.text
        assert any(chunk.reasoning for chunk in chunks)
        assert reply.reasoning is not None
        # the thinking signature arrives in a streamed delta of its own
        assert reply.reasoning.extra["extra_content"]["anthropic"]["signature"]

    @requires_api_key
    @pytest.mark.integration
    def test_live_agent_loop_with_anthropic_reasoning(self, tools):
        generator = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            model=ANTHROPIC_REASONING_MODEL,
            generation_kwargs=ANTHROPIC_GENERATION_KWARGS,
        )
        agent = Agent(chat_generator=generator, tools=tools)

        sent: list[dict] = []
        convert = OtariChatGenerator._convert_message_to_api_format

        def recording_convert(self, message):
            formatted = convert(self, message)
            sent.append(formatted)
            return formatted

        with patch.object(OtariChatGenerator, "_convert_message_to_api_format", recording_convert):
            result = agent.run(messages=[ChatMessage.from_user("What's the weather in Paris? Use the tool.")])

        tool_call_message = next(message for message in result["messages"] if message.tool_calls)
        assert tool_call_message.reasoning is not None
        signature = tool_call_message.reasoning.extra["extra_content"]["anthropic"]["signature"]
        # the next request sends the tool-call reply back with its reasoning and signature
        sent_tool_call_message = next(message for message in sent if message.get("tool_calls"))
        assert sent_tool_call_message["extra_content"]["anthropic"]["signature"] == signature
        assert sent_tool_call_message["reasoning"] == tool_call_message.reasoning.reasoning_text
        assert "32" in result["last_message"].text
