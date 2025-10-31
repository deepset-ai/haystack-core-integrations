import json
import os
import urllib.request
from pathlib import Path
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    ImageContent,
    StreamingChunk,
    TextContent,
    ToolCall,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool, Toolset, create_tool_from_function

from haystack_integrations.components.generators.llama_cpp.chat.chat_generator import (
    LlamaCppChatGenerator,
    _convert_message_to_llamacpp_format,
)


@pytest.fixture
def model_path():
    return Path(__file__).parent / "models"


def get_current_temperature(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]):
    """Get the current temperature in a given location"""

    if "tokyo" in location.lower():
        return {"location": "Tokyo", "temperature": "10", "unit": "celsius"}
    if "san francisco" in location.lower():
        return {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
    if "paris" in location.lower():
        return {"location": "Paris", "temperature": "22", "unit": "celsius"}

    return {"location": location, "temperature": "unknown"}


@pytest.fixture
def temperature_tool():
    return create_tool_from_function(get_current_temperature)


def download_file(file_link, filename, capsys):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)  # noqa: S310
        with capsys.disabled():
            print("\nModel file downloaded successfully.")
    else:
        with capsys.disabled():
            print("\nModel file already exists.")


def test_convert_message_to_llamacpp_format():
    message = ChatMessage.from_system("You are good assistant")
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "system",
        "content": "You are good assistant",
    }

    message = ChatMessage.from_user("I have a question")
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "user",
        "content": "I have a question",
    }

    message = ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "assistant",
        "content": "I have an answer",
    }

    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
    )
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                "id": "123",
            }
        ],
    }

    tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
    message = ChatMessage.from_tool(
        tool_result=tool_result,
        origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}),
    )
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "function",
        "content": tool_result,
        "name": "weather",
    }


def test_convert_message_to_llamacpp_invalid():
    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
    with pytest.raises(ValueError):
        _convert_message_to_llamacpp_format(message)


def test_convert_message_to_llamacpp_format_with_image():
    """Test that a ChatMessage with ImageContent is converted to LlamaCpp format correctly."""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
    message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

    llamacpp_message = _convert_message_to_llamacpp_format(message)

    assert llamacpp_message["role"] == "user"
    assert len(llamacpp_message["content"]) == 2

    # Check text and image parts
    assert llamacpp_message["content"][0]["type"] == "text"
    assert llamacpp_message["content"][0]["text"] == "What's in this image?"
    assert llamacpp_message["content"][1]["type"] == "image_url"
    assert llamacpp_message["content"][1]["image_url"]["url"] == f"data:image/png;base64,{base64_image}"


def test_convert_message_to_llamacpp_format_with_unsupported_mime_type():
    """Test that a ChatMessage with unsupported mime type raises ValueError."""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    image_content = ImageContent(base64_image=base64_image, mime_type="image/bmp")  # Unsupported format
    message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

    with pytest.raises(ValueError, match="Unsupported image format: image/bmp"):
        _convert_message_to_llamacpp_format(message)


def test_convert_message_to_llamacpp_format_with_none_mime_type():
    """Test that a ChatMessage with None mime type raises ValueError."""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
    # Manually set mime_type to None to test the validation
    image_content.mime_type = None
    message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

    with pytest.raises(ValueError, match="Unsupported image format: None"):
        _convert_message_to_llamacpp_format(message)


def test_convert_message_to_llamacpp_format_image_in_non_user_message():
    """Test that images in non-user messages raise ValueError."""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
    message = ChatMessage.from_assistant(text="Here's an image", meta={})
    # Manually add image to assistant message to test constraint
    message._content.append(image_content)

    with pytest.raises(ValueError, match="Image content is only supported for user messages"):
        _convert_message_to_llamacpp_format(message)

    message = ChatMessage(
        _role=ChatRole.ASSISTANT,
        _content=[
            TextContent(text="I have an answer"),
            TextContent(text="I have another answer"),
        ],
    )
    with pytest.raises(ValueError):
        _convert_message_to_llamacpp_format(message)

    tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
    message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
    with pytest.raises(ValueError):
        _convert_message_to_llamacpp_format(message)

    message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
    with pytest.raises(ValueError):
        _convert_message_to_llamacpp_format(message)


def test_handle_streaming_response():
    llama_cpp_chunks = [
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " France"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " is"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " located"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " in"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " Western"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " Europe"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": ","}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " b"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "ordered"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " by"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " Luxem"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "bourg"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": ","}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " Germany"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": ","}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " Switzerland"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1a120f79-e730-4fda-8708-ace395afd03a",
            "model": "tests/models/openchat-3.5-1210.Q3_K_S.gguf",
            "created": 1753457814,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
        },
    ]

    generator = LlamaCppChatGenerator(model="tests/models/openchat-3.5-1210.Q3_K_S.gguf")
    component_info = ComponentInfo.from_component(generator)

    message = generator._handle_streaming_response(llama_cpp_chunks, print_streaming_chunk, component_info)["replies"][
        0
    ]
    assert message.text == " France is located in Western Europe, bordered by Luxembourg, Germany, Switzerland"
    assert message.tool_calls == []
    assert message.meta["finish_reason"] == "stop"
    assert message.meta["model"] == "tests/models/openchat-3.5-1210.Q3_K_S.gguf"
    assert "completion_start_time" in message.meta


def test_handle_streaming_response_tool_calls():
    llama_cpp_chunks = [
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {"role": "assistant", "content": None, "function_call": None, "tool_calls": None},
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": "{"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": "{"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": ' "'},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": ' "'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": "city"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": "city"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": '":'},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": '":'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": ' "'},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": ' "'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": "Tok"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": "Tok"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": "yo"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": "yo"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": '"'},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": '"'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": " }"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": " }"},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": " "},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": " "},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                    "delta": {
                        "role": None,
                        "content": None,
                        "function_call": {"name": "get_current_weather", "arguments": ""},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
                                "type": "function",
                                "function": {"name": "get_current_weather", "arguments": ""},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            "object": "chat.completion.chunk",
            "created": 1753457703,
            "model": "tests/models/functionary-small-v2.4.Q4_0.gguf",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                    "delta": {"role": None, "content": None, "function_call": None, "tool_calls": None},
                }
            ],
        },
    ]

    generator = LlamaCppChatGenerator(model="tests/models/functionary-small-v2.4.Q4_0.gguf")
    component_info = ComponentInfo.from_component(generator)

    message = generator._handle_streaming_response(llama_cpp_chunks, print_streaming_chunk, component_info)["replies"][
        0
    ]

    assert not message.text
    assert message.tool_calls == [
        ToolCall(
            id="call__0_get_current_weather_cmpl-9eb96873-de1f-43af-8b9f-89ac4fdb58e2",
            tool_name="get_current_weather",
            arguments={"city": "Tokyo"},
        )
    ]
    assert message.meta["finish_reason"] == "tool_calls"
    assert message.meta["model"] == "tests/models/functionary-small-v2.4.Q4_0.gguf"
    assert "completion_start_time" in message.meta


class TestLlamaCppChatGenerator:
    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q5_K_S.gguf"
        )
        filename = "Qwen_Qwen3-0.6B-Q5_K_S.gguf"

        # Download GGUF model from HuggingFace
        download_file(gguf_model_path, str(model_path / filename), capsys)

        model_path = str(model_path / filename)
        generator = LlamaCppChatGenerator(model=model_path, n_ctx=8192, n_batch=512)
        generator.warm_up()
        return generator

    @pytest.fixture
    def generator_mock(self):
        mock_model = MagicMock()
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=2048, n_batch=512)
        generator._model = mock_model
        return generator, mock_model

    def test_default_init(self):
        """
        Test default initialization parameters.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf")

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 0
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 0, "n_batch": 512}
        assert generator.generation_kwargs == {}

    def test_custom_init(self):
        """
        Test custom initialization parameters.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf",
            n_ctx=8192,
            n_batch=512,
            streaming_callback=print_streaming_chunk,
        )

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 8192
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 8192, "n_batch": 512}
        assert generator.generation_kwargs == {}
        assert generator.streaming_callback == print_streaming_chunk

    def test_init_with_toolset(self, temperature_tool):
        toolset = Toolset([temperature_tool])
        generator = LlamaCppChatGenerator(model="test_model.gguf", tools=toolset)
        assert generator.tools == toolset

    def test_init_with_mixed_tools(self, temperature_tool):
        """Test initialization with mixed Tool and Toolset objects."""

        def population(city: str):
            """Get population for a given city."""
            return f"The population of {city} is 2.2 million"

        population_tool = create_tool_from_function(population)
        toolset = Toolset([population_tool])

        generator = LlamaCppChatGenerator(model="test_model.gguf", tools=[temperature_tool, toolset])
        assert generator.tools == [temperature_tool, toolset]

    def test_run_with_mixed_tools(self, temperature_tool):
        """Test run method with mixed Tool and Toolset objects."""

        def population(city: str):
            """Get population for a given city."""
            return f"The population of {city} is 2.2 million"

        population_tool = create_tool_from_function(population)
        toolset = Toolset([population_tool])

        generator = LlamaCppChatGenerator(model="test_model.gguf")

        # Mock the model
        mock_model = MagicMock()
        mock_response = {
            "choices": [{"message": {"content": "Generated text"}, "index": 0, "finish_reason": "stop"}],
            "id": "test_id",
            "model": "test_model",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_model.create_chat_completion.return_value = mock_response
        generator._model = mock_model

        generator.run(
            messages=[ChatMessage.from_user("What's the weather in Paris and population of Berlin?")],
            tools=[temperature_tool, toolset],
        )

        # Verify the model was called with the correct tools
        mock_model.create_chat_completion.assert_called_once()
        call_args = mock_model.create_chat_completion.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 2  # Both tools should be flattened

        # Verify tool names
        tool_names = {tool["function"]["name"] for tool in call_args["tools"]}
        assert "get_current_temperature" in tool_names
        assert "population" in tool_names

    def test_init_with_multimodal_params(self):
        """Test initialization with multimodal parameters."""
        generator = LlamaCppChatGenerator(
            model="llava-v1.5-7b-q4_0.gguf",
            chat_handler_name="Llava15ChatHandler",
            model_clip_path="mmproj-model-f16.gguf",
            n_ctx=4096,
        )
        assert generator.model_clip_path == "mmproj-model-f16.gguf"
        assert generator.chat_handler_name == "Llava15ChatHandler"
        assert generator.n_ctx == 4096

    def test_init_validation_clip_path_required(self):
        """Test that model_clip_path is required when chat_handler_name is provided."""
        with pytest.raises(ValueError, match="model_clip_path must be provided when chat_handler_name is specified"):
            LlamaCppChatGenerator(model="llava-v1.5-7b-q4_0.gguf", chat_handler_name="Llava15ChatHandler")

    def test_init_validation_unsupported_handler(self):
        """Test that unsupported chat handler names raise ValueError."""
        with pytest.raises(ValueError, match="Failed to import chat handler 'invalid'"):
            LlamaCppChatGenerator(
                model="llava-v1.5-7b-q4_0.gguf", chat_handler_name="invalid", model_clip_path="mmproj-model-f16.gguf"
            )

    def test_to_dict(self):
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=8192, n_batch=512)
        assert generator.to_dict() == {
            "type": "haystack_integrations.components.generators.llama_cpp.chat.chat_generator.LlamaCppChatGenerator",
            "init_parameters": {
                "model": "test_model.gguf",
                "n_ctx": 8192,
                "n_batch": 512,
                "model_kwargs": {"model_path": "test_model.gguf", "n_ctx": 8192, "n_batch": 512},
                "generation_kwargs": {},
                "tools": None,
                "streaming_callback": None,
                "chat_handler_name": None,
                "model_clip_path": None,
            },
        }

    def test_to_dict_with_multimodal_params(self):
        """Test serialization with multimodal parameters."""
        generator = LlamaCppChatGenerator(
            model="llava-v1.5-7b-q4_0.gguf",
            chat_handler_name="Llava15ChatHandler",
            model_clip_path="mmproj-model-f16.gguf",
            n_ctx=4096,
        )
        data = generator.to_dict()

        assert data["init_parameters"]["model_clip_path"] == "mmproj-model-f16.gguf"
        assert data["init_parameters"]["chat_handler_name"] == "Llava15ChatHandler"
        assert data["init_parameters"]["n_ctx"] == 4096

    def test_to_dict_with_toolset(self, temperature_tool):
        toolset = Toolset([temperature_tool])
        generator = LlamaCppChatGenerator(model="test_model.gguf", tools=toolset)

        data = generator.to_dict()

        assert "tools" in data["init_parameters"]["tools"]["data"]
        assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"

    def test_from_dict_with_toolset(self, temperature_tool):
        toolset = Toolset([temperature_tool])
        generator = LlamaCppChatGenerator(model="test_model.gguf", tools=toolset)
        data = generator.to_dict()

        deserialized_component = LlamaCppChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)

    def test_from_dict(self):
        serialized = {
            "type": "haystack_integrations.components.generators.llama_cpp.chat.chat_generator.LlamaCppChatGenerator",
            "init_parameters": {
                "model": "test_model.gguf",
                "n_ctx": 8192,
                "n_batch": 512,
                "model_kwargs": {"model_path": "test_model.gguf", "n_ctx": 8192, "n_batch": 512},
                "generation_kwargs": {},
                "tools": None,
                "streaming_callback": None,
            },
        }
        deserialized = LlamaCppChatGenerator.from_dict(serialized)
        assert deserialized.model_path == "test_model.gguf"
        assert deserialized.n_ctx == 8192
        assert deserialized.n_batch == 512
        assert deserialized.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 8192, "n_batch": 512}
        assert deserialized.generation_kwargs == {}

    def test_ignores_model_path_if_specified_in_model_kwargs(self):
        """
        Test that model_path is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf",
            n_ctx=8192,
            n_batch=512,
            model_kwargs={"model_path": "other_model.gguf"},
        )
        assert generator.model_kwargs["model_path"] == "other_model.gguf"

    def test_ignores_n_ctx_if_specified_in_model_kwargs(self):
        """
        Test that n_ctx is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=512, n_batch=512, model_kwargs={"n_ctx": 8192})
        assert generator.model_kwargs["n_ctx"] == 8192

    def test_ignores_n_batch_if_specified_in_model_kwargs(self):
        """
        Test that n_batch is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf", n_ctx=8192, n_batch=512, model_kwargs={"n_batch": 1024}
        )
        assert generator.model_kwargs["n_batch"] == 1024

    def test_raises_error_without_warm_up(self):
        """
        Test that the generator raises an error if warm_up() is not called before running.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=512, n_batch=512)
        with pytest.raises(RuntimeError):
            generator.run("What is the capital of China?")

    def test_run_with_empty_message(self, generator_mock):
        """
        Test that an empty message returns an empty list of replies.
        """
        generator, _ = generator_mock
        result = generator.run([])
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 0

    def test_run_with_valid_message(self, generator_mock):
        """
        Test that a valid message returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "id": "unique-id-123",
            "model": "Test Model Path",
            "created": 1715226164,
            "choices": [
                {"index": 0, "message": {"content": "Generated text", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 14, "completion_tokens": 57, "total_tokens": 71},
        }
        mock_model.create_chat_completion.return_value = mock_output
        result = generator.run(messages=[ChatMessage.from_system("Test")])
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)
        assert result["replies"][0].text == "Generated text"
        assert result["replies"][0].role == ChatRole.ASSISTANT

    def test_run_with_generation_kwargs(self, generator_mock):
        """
        Test that a valid message and generation kwargs returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "id": "unique-id-123",
            "model": "Test Model Path",
            "created": 1715226164,
            "choices": [
                {"index": 0, "message": {"content": "Generated text", "role": "assistant"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 14, "completion_tokens": 57, "total_tokens": 71},
        }
        mock_model.create_chat_completion.return_value = mock_output
        generation_kwargs = {"max_tokens": 128}
        result = generator.run([ChatMessage.from_system("Write a 200 word paragraph.")], generation_kwargs)
        assert result["replies"][0].text == "Generated text"
        assert result["replies"][0].meta["finish_reason"] == "length"

    @pytest.mark.integration
    def test_run(self, generator):
        """
        Test that a valid message returns a list of replies.
        """
        questions_and_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        for question, answer in questions_and_answers:
            chat_message = ChatMessage.from_system(
                f"GPT4 Correct User: Answer in a single word. {question} <|end_of_turn|>\n GPT4 Correct Assistant:"
            )
            result = generator.run([chat_message])

            assert "replies" in result
            assert isinstance(result["replies"], list)
            assert len(result["replies"]) > 0
            assert any(answer.lower() in reply.text.lower() for reply in result["replies"])

    @pytest.mark.integration
    def test_run_streaming(self, generator):
        component_info = ComponentInfo.from_component(generator)

        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""
                assert chunk.component_info == component_info

        callback = Callback()

        results = generator.run(
            messages=[ChatMessage.from_user("What's the capital of France?")], streaming_callback=callback
        )

        assert len(results["replies"]) == 1
        assert callback.counter > 0, "No streaming chunks received"
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"

    @pytest.mark.integration
    def test_run_rag_pipeline(self, generator):
        """
        Test that a valid message returns a list of replies.
        """
        document_store = InMemoryDocumentStore()
        documents = [
            Document(content="There are over 7,000 languages spoken around the world today."),
            Document(
                content="""Elephants have been observed to behave in a way that indicates a high
                level of self-awareness, such as recognizing themselves in mirrors."""
            ),
            Document(
                content="""In certain parts of the world, like the Maldives, Puerto Rico,
                and San Diego, you can witness the phenomenon of bioluminescent waves."""
            ),
        ]
        document_store.write_documents(documents=documents)

        pipeline = Pipeline()
        pipeline.add_component(
            instance=InMemoryBM25Retriever(document_store=document_store, top_k=1),
            name="retriever",
        )
        pipeline.add_component(instance=ChatPromptBuilder(variables=["query", "documents"]), name="prompt_builder")
        pipeline.add_component(instance=generator, name="llm")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")

        question = "How many languages are there?"
        location = "Puerto Rico"
        system_message = ChatMessage.from_system(
            "You are a helpful assistant giving out valuable information to tourists."
        )
        messages = [
            system_message,
            ChatMessage.from_user(
                """
        Given these documents and given that I am currently in {{ location }}, answer the question.\nDocuments:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
        """
            ),
        ]
        question = "Can I see bioluminescent waves at my current location?"
        result = pipeline.run(
            data={
                "retriever": {"query": question},
                "prompt_builder": {
                    "template_variables": {"location": location},
                    "template": messages,
                    "query": question,
                },
            }
        )

        replies = result["llm"]["replies"]
        assert len(replies) > 0
        assert any("bioluminescent waves" in reply.text.lower() for reply in replies)
        assert all(reply.role == ChatRole.ASSISTANT for reply in replies)

    @pytest.mark.integration
    def test_json_constraining(self, generator):
        """
        Test that the generator can output valid JSON.
        """
        messages = [ChatMessage.from_system("Output valid json only. List 2 people with their name and age.")]
        json_schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["people"],
        }

        result = generator.run(
            messages=messages,
            generation_kwargs={
                "response_format": {"type": "json_object", "schema": json_schema},
            },
        )

        assert "replies" in result
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in result["replies"])
        for reply in result["replies"]:
            assert json.loads(reply.text)
            assert isinstance(json.loads(reply.text), dict)
            assert "people" in json.loads(reply.text)
            assert isinstance(json.loads(reply.text)["people"], list)
            assert all(isinstance(person, dict) for person in json.loads(reply.text)["people"])
            assert all("name" in person for person in json.loads(reply.text)["people"])
            assert all("age" in person for person in json.loads(reply.text)["people"])
            assert all(isinstance(person["name"], str) for person in json.loads(reply.text)["people"])
            assert all(isinstance(person["age"], int) for person in json.loads(reply.text)["people"])

    def test_multimodal_message_processing(self):
        """Test multimodal message processing with mocked model."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

        # Mock the model
        mock_model = MagicMock()
        mock_response = {
            "choices": [{"message": {"content": "I can see an image."}, "index": 0, "finish_reason": "stop"}],
            "id": "test_id",
            "model": "test_model",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_model.create_chat_completion.return_value = mock_response

        generator = LlamaCppChatGenerator(
            model="test_model.gguf", chat_handler_name="Llava15ChatHandler", model_clip_path="test_clip.gguf"
        )
        generator._model = mock_model

        result = generator.run(messages)

        # Verify the multimodal message was processed correctly
        assert "replies" in result
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "I can see an image."

        # Verify the model was called with the correct format
        mock_model.create_chat_completion.assert_called_once()
        call_args = mock_model.create_chat_completion.call_args[1]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1

        # Check the message format
        llamacpp_message = call_args["messages"][0]
        assert llamacpp_message["role"] == "user"
        assert isinstance(llamacpp_message["content"], list)
        assert len(llamacpp_message["content"]) == 2
        assert llamacpp_message["content"][0]["type"] == "text"
        assert llamacpp_message["content"][1]["type"] == "image_url"


class TestLlamaCppChatGeneratorFunctionary:
    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/meetkai/functionary-small-v2.4-GGUF/resolve/main/functionary-small-v2.4.Q4_0.gguf"
        )
        filename = "functionary-small-v2.4.Q4_0.gguf"
        download_file(gguf_model_path, str(model_path / filename), capsys)
        model_path = str(model_path / filename)
        hf_tokenizer_path = "meetkai/functionary-small-v2.4-GGUF"
        generator = LlamaCppChatGenerator(
            model=model_path,
            n_ctx=512,
            n_batch=512,
            model_kwargs={
                "chat_format": "functionary-v2",
                "hf_tokenizer_path": hf_tokenizer_path,
            },
        )
        generator.warm_up()
        return generator

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming_callback", [None, print_streaming_chunk])
    def test_function_call(self, generator, streaming_callback):
        def get_user_info(username: Annotated[str, "The username to retrieve information for."]):
            """Retrieves detailed information about a user."""
            return {"username": username, "age": 25, "location": "San Francisco"}

        tool = create_tool_from_function(get_user_info)

        tool_choice = {"type": "function", "function": {"name": "get_user_info"}}

        messages = [
            ChatMessage.from_user("Get information for user john_doe"),
        ]
        response = generator.run(
            messages=messages,
            tools=[tool],
            generation_kwargs={"tool_choice": tool_choice},
            streaming_callback=streaming_callback,
        )

        reply = response["replies"][0]

        assert reply.role == ChatRole.ASSISTANT
        assert reply.tool_calls
        tool_calls = reply.tool_calls
        assert len(tool_calls) > 0
        assert tool_calls[0].tool_name == "get_user_info"
        assert tool_calls[0].arguments == {"username": "john_doe"}

    @pytest.mark.integration
    def test_function_call_and_execute(self, generator, temperature_tool):
        user_message = ChatMessage.from_user("What's the weather like in San Francisco?")

        tool_choice = {"type": "function", "function": {"name": "get_current_temperature"}}
        response = generator.run(
            messages=[user_message], tools=[temperature_tool], generation_kwargs={"tool_choice": tool_choice}
        )

        assert "replies" in response
        assert len(response["replies"]) > 0
        first_reply = response["replies"][0]
        assert first_reply.tool_calls
        tool_calls = first_reply.tool_calls

        # tool invocation
        tool_call = tool_calls[0]
        function_args = tool_call.arguments
        tool_response = str(temperature_tool.invoke(**function_args))

        tool_message = ChatMessage.from_tool(tool_result=tool_response, origin=tool_call)

        all_messages = [user_message, first_reply, tool_message]

        second_response = generator.run(messages=all_messages)
        assert "replies" in second_response
        assert len(second_response["replies"]) > 0
        assert any("San Francisco" in reply.text for reply in second_response["replies"])
        assert any("72" in reply.text for reply in second_response["replies"])


class TestLlamaCppChatGeneratorChatML:
    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q5_K_S.gguf"
        )
        filename = "Qwen_Qwen3-0.6B-Q5_K_S.gguf"
        download_file(gguf_model_path, str(model_path / filename), capsys)
        model_path = str(model_path / filename)
        generator = LlamaCppChatGenerator(
            model=model_path,
            n_ctx=8192,
            n_batch=512,
            model_kwargs={
                "chat_format": "chatml-function-calling",
            },
        )
        generator.warm_up()
        return generator

    @pytest.mark.integration
    def test_function_call_chatml(self, generator):
        def get_user_detail(name: Annotated[str, "The name of the user"], age: Annotated[int, "The age of the user"]):
            """Retrieves detailed information about a user."""
            pass

        tool = create_tool_from_function(get_user_detail)

        messages = [
            ChatMessage.from_system(
                """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful,
                detailed, and polite answers to the user's questions. The assistant calls functions with appropriate
                input when necessary"""
            ),
            ChatMessage.from_user("Get details for user: Jason who is 25 years old"),
        ]

        tool_choice = {"type": "function", "function": {"name": "get_user_detail"}}

        response = generator.run(messages=messages, tools=[tool], generation_kwargs={"tool_choice": tool_choice})

        reply = response["replies"][0]
        assert reply.tool_calls
        tool_calls = reply.tool_calls
        assert len(tool_calls) > 0
        assert tool_calls[0].tool_name == "get_user_detail"
        arguments = tool_calls[0].arguments
        assert "name" in arguments
        assert "age" in arguments
        assert arguments["name"] == "Jason"
        assert arguments["age"] == 25

    @pytest.fixture
    def vision_language_model(self, model_path, capsys):
        """Download Vision Language Model for integration testing."""
        # Download text model
        text_model_url = "https://huggingface.co/abetlen/nanollava-gguf/resolve/main/nanollava-text-model-f16.gguf"
        text_model_file = "nanollava-text-model-f16.gguf"
        download_file(text_model_url, str(model_path / text_model_file), capsys)

        # Download vision model
        mmproj_url = "https://huggingface.co/abetlen/nanollava-gguf/resolve/main/nanollava-mmproj-f16.gguf"
        mmproj_file = "nanollava-mmproj-f16.gguf"
        download_file(mmproj_url, str(model_path / mmproj_file), capsys)

        return str(model_path / text_model_file), str(model_path / mmproj_file)

    @pytest.mark.integration
    def test_live_run_image_support(self, vision_language_model):
        text_model_path, mmproj_model_path = vision_language_model

        image_content = ImageContent.from_file_path("tests/test_files/apple.jpg")

        messages = [ChatMessage.from_user(content_parts=["What do you see in this image? Max 5 words.", image_content])]

        generator = LlamaCppChatGenerator(
            model=text_model_path,
            chat_handler_name="NanoLlavaChatHandler",
            model_clip_path=mmproj_model_path,
            n_ctx=2048,
            generation_kwargs={"max_tokens": 50, "temperature": 0.1},
        )

        generator.warm_up()

        result = generator.run(messages)

        assert "replies" in result
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) > 0

        reply = result["replies"][0]
        assert isinstance(reply, ChatMessage)
        assert reply.text is not None
        assert "apple" in reply.text.lower()
