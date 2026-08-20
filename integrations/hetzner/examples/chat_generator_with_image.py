# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# The models served by the Hetzner Inference API accept images alongside text.
# This example demonstrates how to send an image to the HetznerChatGenerator component.
# To run this example, you will need to set the `HETZNER_API_KEY` environment variable.

from haystack.dataclasses import ChatMessage, ImageContent

from haystack_integrations.components.generators.hetzner import HetznerChatGenerator

image = ImageContent.from_url("https://cdn.hetzner.de/cdn/public/Uploads/Finnland_Luftaufnahme-v2.jpg")

messages = [ChatMessage.from_user(content_parts=["Describe this image in one sentence.", image])]

client = HetznerChatGenerator()
response = client.run(messages=messages)

print(response["replies"][0].text)
