# context-dev-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/context-dev-haystack.svg)](https://pypi.org/project/context-dev-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/context-dev-haystack.svg)](https://pypi.org/project/context-dev-haystack)

- [Context.dev documentation](https://docs.context.dev)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/context/CHANGELOG.md)

---

## Installation

```bash
pip install context-dev-haystack
```

Create an API key in the [Context.dev dashboard](https://www.context.dev/dashboard/api-keys) and set it in your environment:

```bash
export CONTEXT_API_KEY="your-api-key"
```

## Usage

Search the live web and receive relevance-ranked Haystack Documents:

```python
from haystack_integrations.components.websearch.context import ContextWebSearch

websearch = ContextWebSearch(top_k=5)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

Fetch known pages as clean Markdown:

```python
from haystack_integrations.components.fetchers.context import ContextFetcher

fetcher = ContextFetcher()
result = fetcher.run(urls=["https://haystack.deepset.ai"])
documents = result["documents"]
```

Crawl a bounded set of pages from a website:

```python
from haystack_integrations.components.fetchers.context import ContextCrawler

crawler = ContextCrawler(crawl_params={"maxPages": 10, "maxDepth": 2})
result = crawler.run(urls=["https://docs.haystack.deepset.ai"])
documents = result["documents"]
```

All three components also provide `run_async`.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The unit tests are fully mocked and need no credentials. Export `CONTEXT_API_KEY` to run the live integration tests.
