# ShopSavvy Haystack Integration

[![PyPI](https://img.shields.io/pypi/v/shopsavvy-haystack.svg)](https://pypi.org/project/shopsavvy-haystack/)

This integration provides [Haystack](https://haystack.deepset.ai/) components for the [ShopSavvy Data API](https://shopsavvy.com/data), enabling product search and price comparison across thousands of retailers.

## Installation

```bash
pip install shopsavvy-haystack
```

Get an API key at [shopsavvy.com/data](https://shopsavvy.com/data).

## Components

### ShopSavvyProductSearch

Search for products by keyword. Returns Haystack Documents with product details.

```python
from haystack_integrations.components.converters.shopsavvy import ShopSavvyProductSearch
from haystack.utils import Secret

search = ShopSavvyProductSearch(
    api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
    top_k=5,
)
result = search.run(query="sony headphones")
for doc in result["documents"]:
    print(doc.meta["title"], doc.meta["brand"])
```

### ShopSavvyPriceComparison

Compare current prices for a product across retailers. Returns one Document per offer.

```python
from haystack_integrations.components.converters.shopsavvy import ShopSavvyPriceComparison
from haystack.utils import Secret

compare = ShopSavvyPriceComparison(
    api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
)
result = compare.run(identifier="B09XS7JWHH")
for doc in result["documents"]:
    print(doc.meta["retailer"], doc.meta["price"], doc.meta["currency"])
```

### Using in a Pipeline

```python
from haystack import Pipeline
from haystack_integrations.components.converters.shopsavvy import (
    ShopSavvyProductSearch,
    ShopSavvyPriceComparison,
)

pipeline = Pipeline()
pipeline.add_component("search", ShopSavvyProductSearch())
pipeline.add_component("compare", ShopSavvyPriceComparison())

# Search then compare
search_result = pipeline.run({"search": {"query": "iphone 15 pro"}})
```

## Development

```bash
# Install with test dependencies
pip install -e ".[dev]"

# Run unit tests
pytest -m "not integration" tests/

# Run integration tests (needs SHOPSAVVY_API_KEY)
export SHOPSAVVY_API_KEY=ss_live_your_key
pytest -m "integration" tests/
```

## License

`shopsavvy-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
