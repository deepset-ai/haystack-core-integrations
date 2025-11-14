# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: T201

"""Run AIMLAPI chat generation with tool calling and execution."""

from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.aimlapi import AIMLAPIChatGenerator


def weather(city: str) -> str:
    """Return mock weather info for the given city."""

    return f"The weather in {city} is sunny and 32°C"


def main() -> None:
    """Demonstrate calling a tool and feeding the result back to AIMLAPI."""

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

    tool_invoker = ToolInvoker(tools=[weather_tool])

    client = AIMLAPIChatGenerator(model="openai/gpt-5-mini-2025-08-07")

    messages = [
        ChatMessage.from_system("You help users by calling the provided tools when they are relevant."),
        ChatMessage.from_user("What's the weather in Tokyo today?"),
    ]

    print("Requesting a tool call from the model...")
    tool_request = client.run(
        messages=messages,
        tools=[weather_tool],
        generation_kwargs={"tool_choice": {"type": "function", "function": {"name": "weather"}}},
    )["replies"][0]

    print(f"assistant tool request: {tool_request}")

    if not tool_request.tool_calls:
        print("No tool call was produced by the model.")
        return

    tool_messages = tool_invoker.run(messages=[tool_request])["tool_messages"]
    for tool_message in tool_messages:
        for tool_result in tool_message.tool_call_results:
            print(f"tool output: {tool_result.result}")

    follow_up_messages = [*messages, tool_request, *tool_messages]

    final_reply = client.run(
        messages=follow_up_messages,
        tools=[weather_tool],
        generation_kwargs={"tool_choice": "none"},
    )["replies"][0]

    print(f"assistant final answer: {final_reply.text}")


if __name__ == "__main__":
    main()
