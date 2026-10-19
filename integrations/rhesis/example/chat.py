import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors.rhesis import RhesisConnector

if __name__ == "__main__":
    pipe = Pipeline()
    pipe.add_component("tracer", RhesisConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(
        data={
            "prompt_builder": {
                "template_variables": {"location": "Berlin"},
                "template": messages,
            },
            "tracer": {"invocation_context": {"session_id": "chat-example"}},
        }
    )
    print(response["llm"]["replies"][0])
    print(response["tracer"]["trace_url"])
    print(response["tracer"]["trace_id"])
