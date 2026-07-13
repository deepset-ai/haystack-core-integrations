# ddgs-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/ddgs-haystack.svg)](https://pypi.org/project/ddgs-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ddgs-haystack.svg)](https://pypi.org/project/ddgs-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/ddgs/CHANGELOG.md)

---

Web search for Haystack backed by [`ddgs`](https://github.com/deedy5/ddgs) ("Dux Distributed Global Search"),
a free, **keyless** metasearch library that aggregates results from multiple backends (DuckDuckGo, Google,
Bing, Brave, Yahoo, Yandex, Mullvad, and more). Unlike a single-engine integration, no API key is required.

## Installation

```bash
pip install ddgs-haystack
```

## Usage

```python
from haystack_integrations.components.websearch.ddgs import DDGSWebSearch

websearch = DDGSWebSearch(top_k=5)
result = websearch.run(query="What is the Haystack framework by deepset?")

for document in result["documents"]:
    print(document.meta["title"], document.meta["url"])
print(result["links"])
```

Pick specific backends, a region, and forward any extra `DDGS().text()` argument via `search_params`:

```python
websearch = DDGSWebSearch(
    top_k=10,
    backend="duckduckgo, brave",
    region="de-de",
    safesearch="off",
    search_params={"timelimit": "w"},  # results from the past week
)
```

### As an agent tool

Wrap it with `ComponentTool` to give an [`Agent`](https://docs.haystack.deepset.ai/docs/agent) keyless web search:

```python
from haystack.tools import ComponentTool
from haystack_integrations.components.websearch.ddgs import DDGSWebSearch

search_tool = ComponentTool(
    component=DDGSWebSearch(top_k=5),
    name="web_search",
    description="Search the web for current information on any topic.",
)
```

## Testing

```bash
hatch run test:unit         # unit tests (no network)
hatch run test:integration  # live tests (hit the web; no API key needed)
```

## License

`ddgs-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
