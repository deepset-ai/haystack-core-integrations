# To run this example, you will need to set a `STACKIT_API_KEY` environment variable.

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.stackit import STACKITChatGenerator

prompt_builder = ChatPromptBuilder()
llm = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")

messages = [ChatMessage.from_user("Question: {{question}} \\n")]

pipeline = Pipeline()
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)

pipeline.connect("prompt_builder.prompt", "llm.messages")

result = pipeline.run({"prompt_builder": {"template_variables": {"question": "Tell me a joke."}, "template": messages}})

print(result)  # noqa: T201
