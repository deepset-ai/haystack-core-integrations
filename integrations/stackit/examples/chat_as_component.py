# To run this example, you will need to set a `STACKIT_API_KEY` environment variable.

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.stackit import STACKITChatGenerator

generator = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")

result = generator.run([ChatMessage.from_user("Tell me a joke.")])
print(result)  # noqa: T201
