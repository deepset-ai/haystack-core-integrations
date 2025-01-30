# To run this example, you will need to set a `ANTHROPIC_API_KEY` environment variable.

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

messages = [
    ChatMessage.from_system("You are a prompt expert who answers questions based on the given documents."),
    ChatMessage.from_user(
        "Here are the documents:\n"
        "{% for d in documents %} \n"
        "    {{d.content}} \n"
        "{% endfor %}"
        "\nAnswer: {{query}}"
    ),
]

rag_pipeline = Pipeline()
rag_pipeline.add_component("fetcher", LinkContentFetcher())
rag_pipeline.add_component("converter", HTMLToDocument())
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["documents"]))
rag_pipeline.add_component(
    "llm",
    AnthropicChatGenerator(
        api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
        streaming_callback=print_streaming_chunk,
    ),
)


rag_pipeline.connect("fetcher", "converter")
rag_pipeline.connect("converter", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "When should we use prompt engineering and when should we fine-tune?"
rag_pipeline.run(
    data={
        "fetcher": {"urls": ["https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview"]},
        "prompt_builder": {"template_variables": {"query": question}, "template": messages},
    }
)
