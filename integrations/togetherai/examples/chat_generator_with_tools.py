# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Use TogetherAIChatGenerator with tools through an Agent.

To run this example, set the `TOGETHER_API_KEY` environment variable.
The Agent handles the tool-calling loop and returns the final answer.
"""

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.togetherai import TogetherAIChatGenerator


# Define a tool that models can call
def weather(city: str) -> str:
    """Return mock weather info for the given city."""
    return f"The weather in {city} is sunny and 32°C"


def main() -> None:
    """Let an Agent answer a question by calling the weather tool."""
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
        chat_generator=TogetherAIChatGenerator(),
        tools=[weather_tool],
        system_prompt="Use the weather tool when answering weather questions.",
    )

    result = agent.run(messages=[ChatMessage.from_user("What's the weather in Tokyo?")])
    print(f"assistant final answer: {result['last_message'].text}")


if __name__ == "__main__":
    main()
