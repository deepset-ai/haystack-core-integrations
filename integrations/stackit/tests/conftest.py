# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction


@pytest.fixture
def chat_completion():
    return ChatCompletion(
        id="foo",
        model="google/gemma-3-27b-it",
        object="chat.completion",
        created=1791467905,
        choices=[
            Choice(
                index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hello world!")
            )
        ],
        usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
    )


@pytest.fixture
def mock_chat_completion(chat_completion):
    with patch("openai.resources.chat.completions.Completions.create", return_value=chat_completion) as mock_create:
        yield mock_create


@pytest.fixture
def mock_async_chat_completion(chat_completion):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=chat_completion,
    ) as mock_create:
        yield mock_create


@pytest.fixture
def reasoning_completion():
    return ChatCompletion(
        id="foo",
        model="openai/gpt-oss-20b",
        object="chat.completion",
        created=1791467905,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="4", reasoning="We need a brief reply. 2+2=4."),
            )
        ],
        usage=CompletionUsage(prompt_tokens=78, completion_tokens=37, total_tokens=115),
    )


@pytest.fixture
def reasoning_chunks():
    # Sampled from GPT-OSS 20B: role, reasoning tokens, answer, finish, then usage.
    deltas = [
        ChoiceDelta(role="assistant", content=""),
        *[
            ChoiceDelta(reasoning=text)
            for text in ["We", " need", " a", " brief", " reply", ".", " ", "2", "+", "2", "=", "4", "."]
        ],
        ChoiceDelta(content="4"),
    ]
    chunks = [
        ChatCompletionChunk(
            id="foo",
            model="openai/gpt-oss-20b",
            object="chat.completion.chunk",
            created=1791467905,
            choices=[ChunkChoice(index=0, delta=delta)],
        )
        for delta in deltas
    ]
    chunks.extend(
        [
            ChatCompletionChunk(
                id="foo",
                model="openai/gpt-oss-20b",
                object="chat.completion.chunk",
                created=1791467905,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            ),
            ChatCompletionChunk(
                id="foo",
                model="openai/gpt-oss-20b",
                object="chat.completion.chunk",
                created=1791467905,
                choices=[],
                usage=CompletionUsage(prompt_tokens=78, completion_tokens=24, total_tokens=102),
            ),
        ]
    )
    return chunks


@pytest.fixture
def reasoning_tool_chunk():
    return ChatCompletionChunk(
        id="foo",
        model="openai/gpt-oss-20b",
        object="chat.completion.chunk",
        created=1791467905,
        choices=[
            ChunkChoice(
                index=0,
                finish_reason="tool_calls",
                delta=ChoiceDelta(
                    reasoning="Check both cities.",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="weather", arguments='{"city":"Paris"}'),
                        ),
                        ChoiceDeltaToolCall(
                            index=1,
                            id="call_2",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="weather", arguments='{"city":"Berlin"}'),
                        ),
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=78, completion_tokens=24, total_tokens=102),
    )
