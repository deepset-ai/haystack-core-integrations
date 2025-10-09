# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.fallback_chat import FallbackChatGenerator


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """Callback function to handle streaming chunks as they arrive."""
    print(chunk.content, end="", flush=True)


def main() -> None:
    # This example uses OpenAI and Anthropic chat generators. You can replace
    # these with any other chat generators from haystack or haystack integrations.
    # Timeout enforcement is delegated to the underlying generators.

    primary = AnthropicChatGenerator(model="claude-sonnet-4-20250514", timeout=10.0)
    backup = OpenAIChatGenerator(model="gpt-4.1-mini", timeout=10.0)

    fallback = FallbackChatGenerator(generators=[primary, backup])

    messages = [ChatMessage.from_user("What's the meaning of life? Think hard about it and answer in elaborate detail.")]

    print("Question: What's the meaning of life? Think hard about it and answer in elaborate detail.")
    print("Streaming response: ", end="")

    result = fallback.run(messages=messages, streaming_callback=print_streaming_chunk)

    print("\n\n--- Streaming complete ---")
    print(f"Successful generator: {result['meta']['successful_generator_class']}")
    print(f"Total attempts: {result['meta']['total_attempts']}")


if __name__ == "__main__":
    main()
