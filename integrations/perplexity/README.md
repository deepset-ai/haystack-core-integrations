# perplexity-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/perplexity-haystack.svg)](https://pypi.org/project/perplexity-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/perplexity-haystack.svg)](https://pypi.org/project/perplexity-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/perplexity/CHANGELOG.md)

---

## Installation

```bash
pip install perplexity-haystack
```

## Usage

Set your `PERPLEXITY_API_KEY` and use `PerplexityWebSearch` in a pipeline:

```python
from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.components.websearch.perplexity import PerplexityWebSearch

websearch = PerplexityWebSearch(
    api_key=Secret.from_env_var("PERPLEXITY_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

See the [Perplexity Search API reference](https://docs.perplexity.ai/api-reference/search-post) for the full list of supported parameters.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `PERPLEXITY_API_KEY` environment variable.
