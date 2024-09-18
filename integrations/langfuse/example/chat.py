import os

# See README.md for more information on how to set up the environment variables
# before running this script

# In addition to setting the environment variables, you need to install the following packages:
# pip install cohere-haystack anthropic-haystack
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from haystack.utils.hf import HFGenerationAPIType

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.cohere import CohereChatGenerator

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

selected_chat_generator = "openai"

generators = {
    "openai": OpenAIChatGenerator,
    "anthropic": AnthropicChatGenerator,
    "hf_api": lambda: HuggingFaceAPIChatGenerator(
        api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
        api_params={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
        token=Secret.from_token(os.environ["HF_API_KEY"]),
    ),
    "cohere": CohereChatGenerator,
}

selected_chat_generator = generators[selected_chat_generator]()

if __name__ == "__main__":

    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", selected_chat_generator)

    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages}})
    print(response["llm"]["replies"][0])
    print(response["tracer"]["trace_url"])
