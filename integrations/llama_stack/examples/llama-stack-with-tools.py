# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# This example demonstrates how to use the LlamaStackChatGenerator component
# with tools.
# To run this example, you will need to
# set up Llama Stack Server through an inference provider and have a model available.
# For a quick start on how to setup server with Ollama, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.llama_stack import LlamaStackChatGenerator


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

client = LlamaStackChatGenerator(
    model="ollama/llama3.2:3b",  # model depends on the inference provider used for the Llama Stack Server.
    api_base_url="http://localhost:8321/v1",
)

# The Agent calls the tool and feeds the result back to the model
agent = Agent(chat_generator=client, tools=[weather_tool])
result = agent.run(messages=[ChatMessage.from_user("What's the weather in Tokyo?")])

print(f"messages: {result['messages']}\n")
print(f"assistant final answer: {result['last_message'].text}")
