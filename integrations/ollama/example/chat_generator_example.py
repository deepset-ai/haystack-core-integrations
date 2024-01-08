from haystack.components.builders import DynamicPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack import Pipeline

from ollama_haystack import OllamaChatGenerator

# no parameter init, we don't use any runtime template variables
prompt_builder = DynamicPromptBuilder()
llm = OllamaChatGenerator(model="orca-mini")

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")
location = "Berlin"
messages = [
    ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
    ChatMessage.from_user("Tell me about {{location}}"),
]
replies = pipe.run(data={"prompt_builder": {"template_variables": {"location": location}, "prompt_source": messages}})

print(replies)
