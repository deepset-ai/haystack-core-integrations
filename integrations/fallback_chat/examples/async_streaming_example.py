# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.fallback_chat import FallbackChatGenerator


async def print_streaming_chunk_async(chunk: StreamingChunk) -> None:
    """Async callback function to handle streaming chunks as they arrive."""
    print(chunk.content, end="", flush=True)


async def main() -> None:
    # This example uses OpenAI and Anthropic chat generators. You can replace
    # these with any other chat generators from haystack or haystack integrations.
    # Timeout enforcement is delegated to the underlying generators.

    primary = OpenAIChatGenerator(model="gpt-4.1-mini", timeout=20.0)
    backup = AnthropicChatGenerator(model="claude-sonnet-4-20250514", timeout=20.0)

    fallback = FallbackChatGenerator(generators=[primary, backup])

    messages = [ChatMessage.from_user("Write a short poem about artificial intelligence.")]

    print("Question: Write a short poem about artificial intelligence.")
    print("Streaming response: ", end="")

    result = await fallback.run_async(messages=messages, streaming_callback=print_streaming_chunk_async)

    print("\n\n--- Streaming complete ---")
    print(f"Successful generator: {result['meta']['successful_generator_class']}")
    print(f"Total attempts: {result['meta']['total_attempts']}")    


if __name__ == "__main__":
    asyncio.run(main())
