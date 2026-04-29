# litellm-haystack

An integration of [LiteLLM](https://github.com/BerriAI/litellm) AI gateway into the [Haystack](https://haystack.deepset.ai/) framework. Routes to 100+ LLM providers (OpenAI, Anthropic, Google, AWS Bedrock, Azure, Cohere, Mistral, Groq, and more) via a single `LiteLLMChatGenerator` component.

## Installation

```bash
pip install litellm-haystack
```

## Usage

```python
from haystack_integrations.components.generators.litellm import LiteLLMChatGenerator
from haystack.dataclasses import ChatMessage

generator = LiteLLMChatGenerator(
    model="anthropic/claude-sonnet-4-20250514",
    generation_kwargs={"max_tokens": 1024},
)

messages = [ChatMessage.from_user("What is Natural Language Processing?")]
result = generator.run(messages=messages)
print(result["replies"][0].text)
```

Model names use LiteLLM format: `provider/model-name`. Provider API keys are read from standard environment variables (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

See https://docs.litellm.ai/docs/providers for the full list of supported providers.
