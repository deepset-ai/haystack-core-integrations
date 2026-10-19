"""
Trace a standalone Haystack `Agent` — no pipeline involved.

`RhesisConnector` is constructed and then never used again: building it installs the tracer
globally, which is all an agent needs. Because there is no pipeline to carry an
`invocation_context` input socket, session and test metadata are attached with
`rhesis_invocation_context` instead.

The trace shows the agent loop: one `ai.agent.invoke` root, an `ai.llm.invoke` span per step, and
an `ai.tool.invoke` span per tool call.
"""

import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import rhesis_invocation_context


def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"It is 18 degrees and cloudy in {city}."


def convert_currency(amount: float, to_currency: str) -> str:
    """Convert an amount from euros to another currency."""
    return f"{amount} EUR is about {amount * 1.08:.2f} {to_currency}."


weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    function=get_weather,
)

currency_tool = Tool(
    name="convert_currency",
    description="Convert an amount in euros to another currency.",
    parameters={
        "type": "object",
        "properties": {"amount": {"type": "number"}, "to_currency": {"type": "string"}},
        "required": ["amount", "to_currency"],
    },
    function=convert_currency,
)


if __name__ == "__main__":
    # Constructing the connector installs the tracer. It is never added to a pipeline.
    RhesisConnector("Agent example")

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=[weather_tool, currency_tool],
        system_prompt="You are a travel assistant. Use the tools available to you.",
    )

    # The pipeline's `invocation_context` socket has no equivalent here, so scope it explicitly.
    # Every span opened inside the block joins this session.
    with rhesis_invocation_context({"session_id": "agent-example"}):
        result = agent.run(
            messages=[ChatMessage.from_user("What is the weather in Berlin, and what is 50 EUR in USD?")]
        )

    print(result["last_message"].text)
