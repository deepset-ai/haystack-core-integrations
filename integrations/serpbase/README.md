# serpbase-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/serpbase-haystack.svg)](https://pypi.org/project/serpbase-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/serpbase-haystack.svg)](https://pypi.org/project/serpbase-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/serpbase/CHANGELOG.md)

---

## Overview

This integration adds a [SerpBase](https://serpbase.dev/) web search component to Haystack.
[SerpBase](https://serpbase.dev/) is a Google Search & SERP API that returns structured JSON
results (organic results, featured snippets, "people also ask" questions, news, images, videos
and Maps data) without requiring you to scrape or parse HTML.

The `SerpBaseWebSearch` component turns a query into a list of `Document` objects and links,
so you can use it anywhere you would use the other web search integrations
(e.g. `SerperDevWebSearch` or `TavilyWebSearch`).

## Installation

```bash
pip install serpbase-haystack
```

## Usage

Set the `SERPBASE_API_KEY` environment variable (get a key at [serpbase.dev](https://serpbase.dev/)),
then:

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.websearch.serpbase import SerpBaseWebSearch

websearch = SerpBaseWebSearch(top_k=10)

prompt_builder = PromptBuilder(
    template="""Answer the query based on the following search results.

    Query: {{ query }}
    Results:
    {% for document in documents %}
      - {{ document.content }}
    {% endfor %}
    """
)

pipeline = Pipeline()
pipeline.add_component("websearch", websearch)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.connect("websearch.documents", "prompt_builder.documents")

result = pipeline.run({"websearch": {"query": "Who is the boyfriend of Olivia Wilde?"}})
```

You can also run a standalone search:

```python
from haystack.utils import Secret

from haystack_integrations.components.websearch.serpbase import SerpBaseWebSearch

websearch = SerpBaseWebSearch(api_key=Secret.from_env_var("SERPBASE_API_KEY"), top_k=5)
result = websearch.run(query="What is Haystack?")
print(result["documents"])
print(result["links"])
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `SERPBASE_API_KEY` environment variable.
