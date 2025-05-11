from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("Whats the weather in Tokyo?")]

client = OpenRouterChatGenerator(generation_kwargs={'provider': {'sort': 'throughput'}, 'max_tokens': 10,
                                 
        })

response = client.run(messages)
print(response)