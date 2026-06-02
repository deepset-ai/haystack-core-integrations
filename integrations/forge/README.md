# forge-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/forge-haystack.svg)](https://pypi.org/project/forge-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/forge-haystack.svg)](https://pypi.org/project/forge-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/forge/CHANGELOG.md)

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

```console
pip install forge-haystack
```

## Usage

[Forge](https://voxell.ai) exposes an OpenAI-compatible embeddings API. This integration provides
`ForgeTextEmbedder` and `ForgeDocumentEmbedder`, which subclass Haystack's built-in OpenAI embedders
and point them at the Forge API base URL (`https://api.voxell.ai/v1`).

Set your API key in the `FORGE_API_KEY` environment variable, then:

```python
from haystack import Document
from haystack_integrations.components.embedders.forge import (
    ForgeDocumentEmbedder,
    ForgeTextEmbedder,
)

text_embedder = ForgeTextEmbedder()
print(text_embedder.run("I love pizza!"))

document_embedder = ForgeDocumentEmbedder()
print(document_embedder.run([Document(content="I love pizza!")]))
```

The default model is `forge-pro`. Other accepted models include `forge-turbo` and `forge-ultra-4k`,
as well as the OpenAI-compatible aliases `text-embedding-3-small`, `text-embedding-3-large`, and
`text-embedding-ada-002`. Forge models support Matryoshka representation learning, so a smaller
output dimensionality can be requested via the `dimensions` parameter.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `FORGE_API_KEY` environment variable.
