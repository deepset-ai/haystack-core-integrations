# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# This example demonstrates how to use the HetznerChatGenerator component to stream a response.
# To run this example, you will need to set the `HETZNER_API_KEY` environment variable.

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.hetzner import HetznerChatGenerator

client = HetznerChatGenerator(streaming_callback=print_streaming_chunk)

messages = [
    ChatMessage.from_system("You are a helpful assistant that answers in one short paragraph."),
    ChatMessage.from_user("What is the Hetzner Inference API?"),
]

response = client.run(messages=messages)

print(f"\n\nusage: {response['replies'][0].meta['usage']}")
