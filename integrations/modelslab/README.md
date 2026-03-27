# modelslab-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/modelslab-haystack.svg)](https://pypi.org/project/modelslab-haystack)
[![Test / modelslab](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/modelslab.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/modelslab.yml)

ModelsLab integration for Haystack.

Provides a ChatGenerator component that uses the ModelsLab API for chat completion.

## Installation

```bash
pip install modelslab-haystack
```

## Usage

### Chat Generation

```python
from haystack_integrations.components.generators.modelslab import ModelsLabChatGenerator
from haystack.dataclasses import ChatMessage

generator = ModelsLabChatGenerator(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_tokens": 500,
    }
)

messages = [
    ChatMessage.from_system("You are a helpful assistant."),
    ChatMessage.from_user("What is Python?"),
]

result = generator.run(messages=messages)
print(result["replies"])
```

### Configuration

The component can be configured with the following parameters:

- `api_key`: Your ModelsLab API key. Can be set via `MODELSLAB_API_KEY` environment variable.
- `model`: The model to use for chat completion. Default: `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `generation_kwargs`: Additional parameters to pass to the API (temperature, max_tokens, etc.)

## Contributing

See the [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md) for more information.
