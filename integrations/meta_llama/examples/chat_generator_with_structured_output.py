# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


# This example demonstrates how to use the MetaLlamaChatGenerator component
# with structured outputs.
# To run this example, you will need to
# set `LLAMA_API_KEY` environment variable

from haystack.dataclasses import ChatMessage
from pydantic import BaseModel

from haystack_integrations.components.generators.meta_llama import MetaLlamaChatGenerator


class NobelPrizeInfo(BaseModel):
    recipient_name: str
    award_year: int
    category: str
    achievement_description: str
    nationality: str


chat_messages = [
    ChatMessage.from_user(
        "In 2021, American scientist David Julius received the Nobel Prize in"
        " Physiology or Medicine for his groundbreaking discoveries on how the human body"
        " senses temperature and touch."
    )
]
component = MetaLlamaChatGenerator(generation_kwargs={"response_format": NobelPrizeInfo})
results = component.run(chat_messages)

# print(results)
