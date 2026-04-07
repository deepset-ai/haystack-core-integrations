# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201

"""Basic text generation example using HPCAIChatGenerator."""

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.hpc_ai import HPCAIChatGenerator


def main() -> None:
    """Generate a response without using any tools."""

    generator = HPCAIChatGenerator()

    messages = [
        ChatMessage.from_system("You are a concise assistant."),
        ChatMessage.from_user("Briefly explain what HPC-AI offers."),
    ]

    reply = generator.run(messages=messages)["replies"][0]

    print(f"assistant response: {reply.text}")


if __name__ == "__main__":
    main()
