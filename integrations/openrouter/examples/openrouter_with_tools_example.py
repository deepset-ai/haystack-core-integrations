# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# This example demonstrates how to use the OpenRouterChatGenerator component
# with tools and model routing.
# To run this example, you will need to
# set `OPENROUTER_API_KEY` environment variable

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator


# Define a tool that models can call
def weather(city: str):
    """Return mock weather info for the given city."""
    return f"The weather in {city} is sunny and 32°C"


tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

weather_tool = Tool(
    name="weather",
    description="Useful for getting the weather in a specific city",
    parameters=tool_parameters,
    function=weather,
)

# We can setup model routing by setting the `model` parameter to `openrouter/auto`
# and providing a list of models to route to.
client = OpenRouterChatGenerator(
    model="openrouter/auto",
    generation_kwargs={
        "models": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
    },
)

# The Agent calls the tool and feeds the result back to the model
agent = Agent(chat_generator=client, tools=[weather_tool])
result = agent.run(messages=[ChatMessage.from_user("What's the weather in Tokyo?")])

print(f"messages: {result['messages']}\n")
print(f"assistant final answer: {result['last_message'].text}")
