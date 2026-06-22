# telnyx-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/telnyx-haystack.svg)](https://pypi.org/project/telnyx-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/telnyx-haystack.svg)](https://pypi.org/project/telnyx-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/telnyx/CHANGELOG.md)

---

## Overview

This integration provides Haystack components for Telnyx Inference:

- `TelnyxChatGenerator` for the OpenAI-compatible chat completions endpoint.
- `TelnyxTextEmbedder` and `TelnyxDocumentEmbedder` for the OpenAI-compatible embeddings endpoint.

Set `TELNYX_API_KEY` before using the components.

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

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, export the `TELNYX_API_KEY` environment variable.
