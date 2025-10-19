# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator

response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "CapitalCity",
        "strict": True,
        "schema": {
            "title": "CapitalCity",
            "type": "object",
            "properties": {
                "city": {"title": "City", "type": "string"},
                "country": {"title": "Country", "type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        },
    },
}

chat_messages = [ChatMessage.from_user("What's the capital of France?")]
component = OpenRouterChatGenerator(generation_kwargs={"response_format": response_schema})
results = component.run(chat_messages)

# print(results)
