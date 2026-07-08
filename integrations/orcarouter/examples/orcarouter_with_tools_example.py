# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201

"""
Run OrcaRouter chat generation with tool calling using the Agent component.

The `Agent` drives the tool-calling loop internally: it asks the model for a tool call, executes the
tool, feeds the result back, and repeats until the model produces a final answer.
"""

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.orcarouter import OrcaRouterChatGenerator


def weather(city: str) -> str:
    """Return mock weather info for the given city."""

    return f"The weather in {city} is sunny and 32°C"


def main() -> None:
    """Let an Agent answer a question by calling the weather tool when relevant."""

    weather_tool = Tool(
        name="weather",
        description="Useful for getting the weather in a specific city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=weather,
    )

    agent = Agent(
        chat_generator=OrcaRouterChatGenerator(model="openai/gpt-4o-mini"),
        tools=[weather_tool],
        system_prompt="You help users by calling the provided tools when they are relevant.",
    )

    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Tokyo today?")])

    print(f"assistant final answer: {result['last_message'].text}")


if __name__ == "__main__":
    main()
