# mrscraper-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/mrscraper-haystack.svg)](https://pypi.org/project/mrscraper-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mrscraper-haystack.svg)](https://pypi.org/project/mrscraper-haystack)

Haystack 2.x components and Agent tools for all 15 MrScraper API operations. The integration preserves upstream JSON
objects, arrays, and scalars as native Python values and returns HTML/plain responses as strings.

## Installation

```bash
pip install mrscraper-haystack
```

Set the API token in the environment. Components use this descriptor by default and resolve it only for a request:

```bash
export MRSCRAPER_API_TOKEN="your-token"
```

```python
from haystack.utils import Secret

api_key = Secret.from_env_var("MRSCRAPER_API_TOKEN")
```

Do not commit tokens or pass them as Pipeline or Agent inputs.

## Components

- **Account:** `MrScraperGetAccountInfo`
- **Discovery:** `MrScraperCrawlWebsiteUrls`, `MrScraperSearchGoogleSerp`
- **Extraction:** `MrScraperExtractPageByPrompt`, `MrScraperExtractListings`,
  `MrScraperExtractStructuredData`, `MrScraperFetchRenderedHtml`
- **Results:** `MrScraperGetResults`, `MrScraperGetLatestResults`, `MrScraperGetResultDetail`
- **Scraper Creation:** `MrScraperCreatePromptScraper`, `MrScraperCreateListingScraper`,
  `MrScraperCreateWebsiteCrawlScraper`
- **Scraper Runs:** `MrScraperRunExistingScraper`, `MrScraperRunExistingScraperBatch`

Every component exposes synchronous `run()` and true asynchronous `run_async()` methods and one stable `result`
output. No response fields are invented or converted to Haystack `Document` objects.

## Direct component use

```python
from haystack_integrations.components.mrscraper import MrScraperExtractPageByPrompt

extractor = MrScraperExtractPageByPrompt()
response = extractor.run(
    url="https://example.com/product/1",
    prompt="Extract the product name and price",
    output_schema={"name": "string", "price": "number"},
)
native_result = response["result"]
```

An `output_schema` is compact-JSON encoded and appended once to the operation's prompt. It is not sent as an
independent API field.

## Pipeline use

```python
from haystack import Pipeline
from haystack_integrations.components.mrscraper import MrScraperSearchGoogleSerp

pipeline = Pipeline()
pipeline.add_component("search", MrScraperSearchGoogleSerp())
result = pipeline.run({"search": {"query": "Haystack agents", "format": "json"}})
print(result["search"]["result"])
```

## ComponentTool, Toolset, and Agent use

A single component can be exposed independently:

```python
from haystack.tools import ComponentTool
from haystack_integrations.components.mrscraper import MrScraperGetLatestResults

latest_results = ComponentTool(
    component=MrScraperGetLatestResults(),
    name="mrscraper_get_latest_results",
    description="Get only the newest N results for a scraper.",
)
```

`MrScraperToolset` supplies 15 distinct tools by default. JSON values are compactly and deterministically serialized
at the Agent tool boundary; HTML/plain strings pass through unchanged. Pipeline component outputs remain native.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.tools.mrscraper import MrScraperToolset

tools = MrScraperToolset()
agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=tools)
```

Select one or more documented groups when the Agent needs a smaller surface:

```python
tools = MrScraperToolset(groups=["discovery", "extraction"])
```

Available group values are `account`, `discovery`, `extraction`, `results`, `scraper_creation`, and `scraper_runs`.

## Serialization and secrets

Components, Pipelines, and `MrScraperToolset` serialize environment-variable secret descriptors without resolving the
token. `Secret.from_token(...)` works for direct runtime use but is intentionally not serializable; use
`Secret.from_env_var(...)` for saved Pipelines and Toolsets.

The integration contacts only these fixed service origins:

- `https://api.app.mrscraper.com`
- `https://sync.scraper.mrscraper.com`
- `https://api.mrscraper.com`

Connection and read timeouts are finite and configurable only in component/Toolset constructors. API credentials,
origins, and arbitrary headers are never Agent tool arguments.

## Operational and legal notes

Crawl, rendered-page, listing, and batch operations can take substantial time and consume significant MrScraper
credits. Start with conservative page limits and review costs before running large batches.

Only scrape content you are authorized to access. Review the target site's terms and applicable laws, particularly
before automating access to login-protected or personal data.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
