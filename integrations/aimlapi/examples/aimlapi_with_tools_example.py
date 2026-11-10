# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201

"""Run AIMLAPI chat generation with tool calling and execution."""

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.aimlapi import AIMLAPIChatGenerator


def weather(city: str) -> str:
    """Return mock weather info for the given city."""

    return f"The weather in {city} is sunny and 32°C"


def main() -> None:
    """Demonstrate an Agent calling a tool and feeding the result back to AIMLAPI."""

    tool_parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    weather_tool = Tool(
        name="weather",
        description="Useful for getting the weather in a specific city",
        parameters=tool_parameters,
        function=weather,
    )

    agent = Agent(
        chat_generator=AIMLAPIChatGenerator(model="openai/gpt-5-mini-2025-08-07"),
        system_prompt="You help users by calling the provided tools when they are relevant.",
        tools=[weather_tool],
    )

    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Tokyo today?")])

    for message in result["messages"]:
        if message.tool_call_result:
            print(f"tool output: {message.tool_call_result.result}")

    print(f"assistant final answer: {result['last_message'].text}")


if __name__ == "__main__":
    main()
