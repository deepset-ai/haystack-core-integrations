import os

os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret

from haystack_integrations.components.others.langfuse import LangfuseComponent


if __name__ == "__main__":

    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseComponent("Chat example"))
    pipe.add_component("prompt_builder", DynamicChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(
        data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "prompt_source": messages}}
    )
    print(response["llm"]["replies"][0])
