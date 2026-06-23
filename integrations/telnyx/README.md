# telnyx-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/telnyx-haystack.svg)](https://pypi.org/project/telnyx-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/telnyx-haystack.svg)](https://pypi.org/project/telnyx-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/telnyx/CHANGELOG.md)

---

## Installation

```bash
pip install telnyx-haystack
```

## Overview

This integration provides Haystack components for Telnyx Inference:

- `TelnyxChatGenerator` for the OpenAI-compatible chat completions endpoint.
- `TelnyxTextEmbedder` and `TelnyxDocumentEmbedder` for the OpenAI-compatible embeddings endpoint.

Model availability can vary by account and over time. Use Telnyx's OpenAI-compatible models endpoint for the
current list available to your account.

Set `TELNYX_API_KEY` before using the components.

```bash
export TELNYX_API_KEY="your-telnyx-api-key"
```

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.telnyx import TelnyxChatGenerator

generator = TelnyxChatGenerator()
result = generator.run([ChatMessage.from_user("Tell me about Telnyx Inference.")])
print(result["replies"][0].text)
```

```python
from haystack_integrations.components.embedders.telnyx import TelnyxTextEmbedder

embedder = TelnyxTextEmbedder()
result = embedder.run("I love pizza!")
print(result["embedding"])
```

Telnyx's current `thenlper/gte-large` embedding model returns fixed-size embeddings, so leave `dimensions` unset
when using the default embedding model. For reasoning-heavy chat models such as `zai-org/GLM-5.1-FP8`, use a
large enough `max_tokens` value for the model to return visible text.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, export the `TELNYX_API_KEY` environment variable.
