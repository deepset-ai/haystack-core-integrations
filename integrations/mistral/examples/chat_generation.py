from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = MistralChatGenerator(model= "mistral-small", streaming_callback=print_streaming_chunk)
response = client.run(messages)
print(response)