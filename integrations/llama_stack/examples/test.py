# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# This example demonstrates how to use the LlamaStackChatGenerator component
# with tools and model routing.
# To run this example, you will need to
# start the llama-stack server

from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.generators.llama_stack.chat.chat_generator import LlamaStackChatGenerator


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

# Create a tool invoker with the weather tool
tool_invoker = ToolInvoker(tools=[weather_tool])

client = LlamaStackChatGenerator(
    api_base_url="http://localhost:8321/v1/openai/v1",
    model="llama3.2:3b",
)
messages = [ChatMessage.from_user("What's the weather in Tokyo?")]

response = client.run(messages=messages, tools=[weather_tool])["replies"]

print(f"assistant messages: {response[0]}\n")  # noqa: T201

# If the assistant message contains a tool call, run the tool invoker
if response[0].tool_calls:
    tool_messages = tool_invoker.run(messages=response)["tool_messages"]
    print(f"tool messages: {tool_messages}")  # noqa: T201
