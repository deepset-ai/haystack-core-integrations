# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


# This example demonstrates how to use the NvidiaChatGenerator component
# with structured outputs.
# To run this example, you will need to
# set `NVIDIA_API_KEY` environment variable

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator

json_schema = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "rating": {"type": "number"}},
    "required": ["title", "rating"],
}
chat_messages = [
    ChatMessage.from_user(
        """
    Return the title and the rating based on the following movie review according
    to the provided json schema.
    Review: Inception is a really well made film. I rate it four stars out of five."""
    )
]

component = NvidiaChatGenerator(
    model="meta/llama-3.1-70b-instruct",
    generation_kwargs={"extra_body": {"nvext": {"guided_json": json_schema}}},
)
results = component.run(chat_messages)
# print(results)
