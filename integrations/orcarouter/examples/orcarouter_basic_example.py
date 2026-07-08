# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201

"""Basic text generation example using OrcaRouterChatGenerator."""

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.orcarouter import OrcaRouterChatGenerator


def main() -> None:
    """Generate a response without using any tools."""

    generator = OrcaRouterChatGenerator(model="openai/gpt-4o-mini")

    messages = [
        ChatMessage.from_system("You are a concise assistant."),
        ChatMessage.from_user("Briefly explain what OrcaRouter offers."),
    ]

    reply = generator.run(messages=messages)["replies"][0]

    print(f"assistant response: {reply.text}")


if __name__ == "__main__":
    main()
