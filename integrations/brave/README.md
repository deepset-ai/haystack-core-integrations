# brave-search-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/brave-search-haystack.svg)](https://pypi.org/project/brave-search-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/brave-search-haystack.svg)](https://pypi.org/project/brave-search-haystack)

Haystack integration for [Brave Search API](https://brave.com/search/api/).

## Installation

```bash
pip install brave-search-haystack
```

## Usage

```python
from haystack_integrations.components.websearch.brave import BraveWebSearch
from haystack.utils import Secret

websearch = BraveWebSearch(
    api_key=Secret.from_env_var("BRAVE_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

## License

`brave-search-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
