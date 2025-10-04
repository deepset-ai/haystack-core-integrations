# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.fallback_chat import FallbackChatGenerator


def main() -> None:
    # This example uses OpenAI and Anthropic chat generators. You can replace
    # these with any other chat generators from haystack or haystack integrations.

    primary = OpenAIChatGenerator(model="gpt-4.1-mini")
    backup = AnthropicChatGenerator(model="claude-sonnet-4-20250514")

    fallback = FallbackChatGenerator(generators=[backup, primary], timeout=10.0)

    messages = [ChatMessage.from_user("Hello! Who are you?")]
    result = fallback.run(messages)
    print(result["replies"][0].text)


if __name__ == "__main__":
    main()
