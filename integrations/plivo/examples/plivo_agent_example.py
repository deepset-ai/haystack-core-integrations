# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Haystack Agent with Plivo tools.

Gives an Agent the full PlivoToolset (send_sms, send_verification_code,
validate_verification_code, lookup_number, make_call) so it can look up a number
and message or verify it, all through the same Plivo account.

Requirements:
    pip install plivo-haystack openai

Environment variables:
    PLIVO_AUTH_ID    - your Plivo Auth ID
    PLIVO_AUTH_TOKEN - your Plivo Auth Token
    OPENAI_API_KEY   - your OpenAI API key  (or swap the generator below)
"""

import sys

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.plivo import PlivoToolset

SENDER = "+14150000000"  # replace with your Plivo number

EXAMPLE_QUERIES = [
    "Look up the number +14151234567 and tell me its carrier and whether it is a mobile line.",
    "Send an SMS to +14151234567 that says 'Your order has shipped.'",
    "Send a verification code to +14151234567 over SMS.",
]


def run(query: str, model: str = "gpt-4o-mini") -> None:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model=model),
        tools=PlivoToolset(sender=SENDER),
        system_prompt=(
            "You are a helpful assistant that can send SMS, run phone verification, "
            "look up phone numbers, and place calls through Plivo. Use the tools when "
            "the user asks for messaging, verification, or number information."
        ),
        max_agent_steps=10,
    )

    result = agent.run(messages=[ChatMessage.from_user(query)])
    print("\n--- Agent response ---")
    print(result["last_message"].text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        run(EXAMPLE_QUERIES[idx])
    else:
        for query in EXAMPLE_QUERIES:
            run(query)
