# mrscraper-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/mrscraper-haystack.svg)](https://pypi.org/project/mrscraper-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mrscraper-haystack.svg)](https://pypi.org/project/mrscraper-haystack)

Haystack 2.x components and Agent tools for all 15 MrScraper API operations. The integration preserves upstream JSON
objects, arrays, and scalars as native Python values and returns HTML/plain responses as strings.

## Installation

```bash
pip install mrscraper-haystack
```

See the [MrScraper API documentation](https://docs.mrscraper.com/) for details about the available API operations.
To create a MrScraper account and get an API token, visit [mrscraper.com](https://mrscraper.com/).

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

## Running examples from this repository

Run commands from the integration directory so Hatch uses this package and its development environment:

```bash
cd integrations/mrscraper
export MRSCRAPER_API_TOKEN="your-token"
hatch run python your_script.py
```

All components accept these constructor parameters:

| Parameter | Default | Description |
|---|---:|---|
| `api_key` | `Secret.from_env_var("MRSCRAPER_API_TOKEN")` | MrScraper API token. Prefer the environment-variable secret for serializable Pipelines and Toolsets. |
| `connect_timeout` | `10.0` | Maximum number of seconds to establish the HTTP connection. Must be a finite number greater than zero. |
| `read_timeout` | `300.0` | Maximum number of seconds to read the HTTP response. Must be a finite number greater than zero. |

Every `run()` call returns `{"result": ...}`. JSON objects, arrays, numbers, booleans, and null values remain native
Python values. HTML and plain-text responses remain exact strings. The examples below print only the value inside
`result`.

## Component reference and examples

The examples show every available `run()` parameter. Remove optional parameters you do not need. For conditional
options on `MrScraperRunExistingScraper`, prefer omitting an option or leaving it as `None` so the MrScraper API can
apply its own default.

### Account

#### `MrScraperGetAccountInfo`

Retrieves the account details, token usage, and token limits associated with the configured API token. `run()` has no
parameters.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperGetAccountInfo

response = MrScraperGetAccountInfo().run()
pprint(response["result"])
```

The response wrapper has this shape. Fields inside `result` are returned exactly as supplied by MrScraper and can
change with the upstream account response:

```python
{
    "result": {
        # Account, subscription, token-usage, and token-limit fields from MrScraper.
    }
}
```

### Discovery

#### `MrScraperCrawlWebsiteUrls`

Discovers URLs immediately by crawling links from a starting website.

| Parameter | Default | Description |
|---|---:|---|
| `url` | Required | Nonblank starting URL. |
| `max_depth` | `2` | Maximum link depth to evaluate. |
| `max_pages` | `50` | Maximum number of pages to evaluate. |
| `limit` | `50` | Maximum number of discovered URLs to return; must be at least `1`. |
| `include_patterns` | `None` | Optional pipe-separated regular expressions for URLs to include. |
| `exclude_patterns` | `None` | Optional pipe-separated regular expressions for URLs to exclude. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperCrawlWebsiteUrls

response = MrScraperCrawlWebsiteUrls().run(
    url="https://example.com",
    max_depth=2,
    max_pages=25,
    limit=25,
    include_patterns=r"/products/|/categories/",
    exclude_patterns=r"/login|/account|/privacy",
)
pprint(response["result"])
```

#### `MrScraperSearchGoogleSerp`

Runs a synchronous Google search through MrScraper.

| Parameter | Default | Description |
|---|---:|---|
| `query` | Required | Nonblank Google search query. |
| `region` | `"us"` | Two-letter country or region code. |
| `language` | `"en"` | Two-letter result-language code. |
| `page` | `1` | Search results page; must be at least `1`. |
| `format` | `"json"` | Output format: `"json"` for native Python data or `"html"` for exact HTML text. |
| `render_js` | `False` | Render JavaScript before collecting the search results. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperSearchGoogleSerp

response = MrScraperSearchGoogleSerp().run(
    query="Haystack AI framework",
    region="id",
    language="en",
    page=1,
    format="json",
    render_js=False,
)
pprint(response["result"])
```

### Extraction

#### `MrScraperExtractPageByPrompt`

Extracts structured information from one page with a General AI scraper.

| Parameter | Default | Description |
|---|---:|---|
| `url` | Required | Nonblank page URL. |
| `prompt` | `None` | Optional natural-language extraction instructions. |
| `output_schema` | `None` | Optional dictionary describing the expected JSON shape. It is appended to the prompt, not sent as a separate API field. |
| `mode` | `"Super"` | MrScraper mode: `"Super"` or `"Cheap"`. |
| `proxy_country` | `None` | Optional two-letter proxy country code. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperExtractPageByPrompt

response = MrScraperExtractPageByPrompt().run(
    url="https://example.com/product/1",
    prompt="Extract the product name, price, and description",
    output_schema={
        "name": "string",
        "price": "number",
        "description": "string",
    },
    mode="Cheap",
    proxy_country="us",
)
pprint(response["result"])
```

#### `MrScraperExtractListings`

Extracts repeated items from a listing or paginated page.

| Parameter | Default | Description |
|---|---:|---|
| `url` | Required | Nonblank starting URL. |
| `prompt` | `None` | Optional instructions describing each listing item. |
| `output_schema` | `None` | Optional dictionary describing each item's expected JSON shape. |
| `max_pages` | `1` | Maximum number of pagination pages to scrape; must be at least `1`. |
| `proxy_country` | `None` | Optional two-letter proxy country code. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperExtractListings

response = MrScraperExtractListings().run(
    url="https://example.com/products",
    prompt="Extract every product shown in the listing",
    output_schema={
        "name": "string",
        "price": "number",
        "url": "string",
    },
    max_pages=3,
    proxy_country="us",
)
pprint(response["result"])
```

#### `MrScraperExtractStructuredData`

Extracts a known data type using a bundled MrScraper prompt preset.

| Parameter | Default | Description |
|---|---:|---|
| `url` | Required | Nonblank page URL. |
| `category` | `"article"` | One of `article`, `forumThread`, `hotel`, `jobPosting`, `post`, `product`, `property`, `restaurant`, `socialMediaProfile`, or `tourAttraction`. |
| `mode` | `"Super"` | MrScraper mode: `"Super"` or `"Cheap"`. |
| `proxy_country` | `None` | Optional two-letter proxy country code. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperExtractStructuredData

response = MrScraperExtractStructuredData().run(
    url="https://example.com/product/1",
    category="product",
    mode="Cheap",
    proxy_country="us",
)
pprint(response["result"])
```

#### `MrScraperFetchRenderedHtml`

Fetches a page with MrScraper's JavaScript-capable stealth browser. Use `markdown=True` when you need Markdown. If
you need only Markdown, you can also set `html=False`. For sites that block direct visits to deep links,
`home_page=True` makes the browser visit the site's home page first and can improve reliability. It may increase run
time and is not a guarantee that a site will allow scraping. `super_mode=True` enables a real device for stronger
scraping capabilities.

| Parameter | Default | Description |
|---|---:|---|
| `url` | Required | Nonblank target URL. |
| `max_retries` | `3` | Maximum upstream retry attempts; must be at least `0`. |
| `timeout` | `300` | Page-load timeout in seconds; must be at least `1`. This is separate from the component's `read_timeout`. |
| `geo_code` | `"us"` | Two-letter geolocation code. |
| `proxy_country` | `"us"` | Two-letter proxy country code. |
| `screenshot` | `False` | Capture a screenshot. |
| `screenshot_mode` | `None` | Screenshot coverage: `"full"` or `"top"`. Required when `screenshot=True`. |
| `html` | `True` | Include rendered HTML. |
| `markdown` | `False` | Include converted Markdown. |
| `token_cap` | `None` | Optional maximum processing-token allowance; must be at least `1`. |
| `wait_for_selector` | `None` | Optional CSS selector to wait for before returning. |
| `wait_until` | `None` | Browser lifecycle event: `"domcontentloaded"`, `"load"`, or `"networkidle"`. |
| `block_resources` | `False` | Block images, fonts, and stylesheets when enabled. |
| `home_page` | `False` | Visit the site's home page before the target URL. Useful when direct deep-link navigation is blocked. |
| `return_cookie` | `False` | Return browser cookies when enabled. |
| `super_mode` | `False` | Use a real device for stronger scraping capabilities. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperFetchRenderedHtml

response = MrScraperFetchRenderedHtml().run(
    url="https://example.com/dynamic-page",
    max_retries=3,
    timeout=300,
    geo_code="us",
    proxy_country="us",
    screenshot=True,
    screenshot_mode="full",
    html=True,
    markdown=True,
    token_cap=10_000,
    wait_for_selector="main",
    wait_until="networkidle",
    block_resources=False,
    home_page=True,
    return_cookie=True,
    super_mode=True,
)
pprint(response["result"])
```

For a simpler Markdown-only request:

```python
response = MrScraperFetchRenderedHtml().run(
    url="https://example.com/dynamic-page",
    html=False,
    markdown=True,
    home_page=True,
)
```

### Results

Scraper runs may finish asynchronously. If a run response does not yet contain the final extracted data, use the
result components below after the run has been accepted.

#### `MrScraperGetResults`

Returns one explicitly paginated and sorted page of results for a scraper.

| Parameter | Default | Description |
|---|---:|---|
| `scraper_id` | Required | Nonblank scraper ID. |
| `page` | `1` | Integer page number. |
| `page_size` | `10` | Integer number of results per page. |
| `sort_by` | `"createdAt"` | Sort field; currently only `"createdAt"`. |
| `sort_order` | `"DESC"` | Sort direction: `"ASC"` or `"DESC"`. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperGetResults

response = MrScraperGetResults().run(
    scraper_id="your-scraper-id",
    page=1,
    page_size=10,
    sort_by="createdAt",
    sort_order="DESC",
)
pprint(response["result"])
```

#### `MrScraperGetLatestResults`

Returns only the newest results for a scraper.

| Parameter | Default | Description |
|---|---:|---|
| `scraper_id` | Required | Nonblank scraper ID. |
| `count` | `10` | Integer number of newest results to request. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperGetLatestResults

response = MrScraperGetLatestResults().run(
    scraper_id="your-scraper-id",
    count=5,
)
pprint(response["result"])
```

#### `MrScraperGetResultDetail`

Returns one complete result. Use the result ID, not the scraper ID.

| Parameter | Default | Description |
|---|---:|---|
| `result_id` | Required | Nonblank result ID. It is safely URL-encoded as one path segment. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperGetResultDetail

response = MrScraperGetResultDetail().run(result_id="your-result-id")
pprint(response["result"])
```

### Scraper creation

#### `MrScraperCreatePromptScraper`

Creates a reusable General AI scraper from extraction instructions. Its parameters have the same meanings as
`MrScraperExtractPageByPrompt`.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperCreatePromptScraper

response = MrScraperCreatePromptScraper().run(
    url="https://example.com/product/1",
    prompt="Extract the product name and price",
    output_schema={"name": "string", "price": "number"},
    mode="Cheap",
    proxy_country="us",
)
pprint(response["result"])
```

Parameters: required `url`; optional `prompt=None`, `output_schema=None`, `mode="Super"` (`"Super"` or `"Cheap"`),
and two-letter `proxy_country=None`.

#### `MrScraperCreateListingScraper`

Creates a reusable Listing AI scraper.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperCreateListingScraper

response = MrScraperCreateListingScraper().run(
    url="https://example.com/products",
    prompt="Extract every product",
    output_schema={"name": "string", "price": "number", "url": "string"},
    max_pages=3,
    proxy_country="us",
)
pprint(response["result"])
```

Parameters: required `url`; optional `prompt=None`, `output_schema=None`, `max_pages=1` (minimum `1`), and two-letter
`proxy_country=None`.

#### `MrScraperCreateWebsiteCrawlScraper`

Creates a reusable Map AI scraper for URL discovery.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperCreateWebsiteCrawlScraper

response = MrScraperCreateWebsiteCrawlScraper().run(
    url="https://example.com",
    max_depth=2,
    max_pages=25,
    limit=25,
    include_patterns=r"/products/|/categories/",
    exclude_patterns=r"/login|/account|/privacy",
)
pprint(response["result"])
```

Parameters: required `url`; optional `max_depth=2`, `max_pages=50`, `limit=50` (minimum `1`),
`include_patterns=None`, and `exclude_patterns=None`. Pattern values are pipe-separated regular expressions.

### Running existing scrapers

#### `MrScraperRunExistingScraper`

Runs one URL with an existing Manual or AI scraper. The valid configuration depends first on `scraper_type`. When
`scraper_type="ai"`, it depends again on `agent_type`.

Parameters shared by every scraper type:

| Parameter | Default | Description |
|---|---:|---|
| `scraper_type` | Required | `"manual"` or `"ai"`. |
| `scraper_id` | Required | Nonblank ID of the existing scraper. |
| `url` | Required | Nonblank URL to process. |
| `max_retry` | `3` | Maximum retry attempts; must be at least `0`. |
| `proxy_country` | `None` | Optional two-letter proxy country code. |

Optional parameters default to `None` and are omitted from the API payload. Do not pass options from a different
branch; the component rejects incompatible combinations before making an HTTP request.

##### Manual scraper

Use `scraper_type="manual"` and do not provide `agent_type`.

| Manual-only or Manual-compatible parameter | Description |
|---|---|
| `timeout` | Request timeout; minimum `1`. |
| `bypass_proxy` | Whether to bypass the configured proxy. |
| `html` | Include HTML. |
| `markdown` | Include Markdown. |
| `screenshot` | Capture a screenshot. |
| `stream` | Stream results. |
| `cookie_jar` | Cookie-jar identifier or serialized value. |
| `cookies` | List of browser-cookie dictionaries. |
| `home_page` | Visit the home page first. |
| `home_page_timeout` | Home-page timeout; minimum `1`. |
| `paginator` | Manual pagination configuration dictionary. |
| `proxy` | Manual proxy URL. |
| `record` | Record the browser session. |
| `return_cookie` | Return browser cookies. |
| `token_cap` | Result token cap; minimum `0`. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraper

response = MrScraperRunExistingScraper().run(
    scraper_type="manual",
    scraper_id="your-manual-scraper-id",
    url="https://example.com/product/1",
    max_retry=3,
    proxy_country="us",
    timeout=300,
    bypass_proxy=False,
    html=True,
    markdown=True,
    screenshot=False,
    stream=False,
    cookie_jar=None,
    cookies=[
        {"name": "locale", "value": "en", "domain": "example.com", "path": "/"},
    ],
    home_page=True,
    home_page_timeout=60,
    paginator=None,
    proxy=None,
    record=False,
    return_cookie=True,
    token_cap=10_000,
)
pprint(response["result"])
```

##### General AI scraper

Use `scraper_type="ai"` and `agent_type="general"`. If `agent_type` is omitted during direct component use, it
defaults to `"general"`.

General AI accepts `bypass_proxy`, `html`, `markdown`, `render_javascript`, `return_cookies`, `screenshot`,
`use_home_page`, and `wait_for_selector`. It rejects Listing-only `max_pages`, `timeout`, and `stream`, all Map-only
options, and all Manual-only options.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraper

response = MrScraperRunExistingScraper().run(
    scraper_type="ai",
    scraper_id="your-general-ai-scraper-id",
    url="https://example.com/product/1",
    max_retry=3,
    proxy_country="us",
    agent_type="general",
    bypass_proxy=False,
    html=True,
    markdown=True,
    render_javascript=True,
    return_cookies=True,
    screenshot=False,
    use_home_page=True,
    wait_for_selector="main",
)
pprint(response["result"])
```

##### Listing AI scraper

Use `scraper_type="ai"` and `agent_type="listing"`. Listing AI accepts all General AI browser options plus:

| Listing-only parameter | Description |
|---|---|
| `max_pages` | Maximum pagination pages; minimum `1`. |
| `timeout` | Listing timeout; minimum `1`. |
| `stream` | Stream listing results. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraper

response = MrScraperRunExistingScraper().run(
    scraper_type="ai",
    scraper_id="your-listing-ai-scraper-id",
    url="https://example.com/products",
    max_retry=3,
    proxy_country="us",
    agent_type="listing",
    max_pages=3,
    timeout=300,
    bypass_proxy=False,
    html=True,
    markdown=True,
    render_javascript=True,
    return_cookies=True,
    screenshot=False,
    stream=False,
    use_home_page=True,
    wait_for_selector=".product-list",
)
pprint(response["result"])
```

##### Map AI scraper

Use `scraper_type="ai"` and `agent_type="map"`. Map AI accepts only the shared parameters and these Map options:

| Map parameter | Description |
|---|---|
| `max_pages` | Maximum pages to evaluate; minimum `1`. |
| `max_depth` | Maximum crawl depth; minimum `0`. |
| `limit` | Maximum discovered URLs to return; minimum `1`. |
| `include_patterns` | Optional pipe-separated regular expressions for URLs to include. |
| `exclude_patterns` | Optional pipe-separated regular expressions for URLs to exclude. |

Map AI rejects browser-output options such as `html`, `markdown`, and `screenshot`, and also rejects `timeout`,
`stream`, `wait_for_selector`, and all Manual-only options.

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraper

response = MrScraperRunExistingScraper().run(
    scraper_type="ai",
    scraper_id="your-map-ai-scraper-id",
    url="https://example.com",
    max_retry=3,
    proxy_country="us",
    agent_type="map",
    max_pages=25,
    max_depth=2,
    limit=25,
    include_patterns=r"/products/|/categories/",
    exclude_patterns=r"/login|/account|/privacy",
)
pprint(response["result"])
```

Valid parameter combinations at a glance:

| Configuration | Additional valid parameters |
|---|---|
| `scraper_type="manual"` | `timeout`, `bypass_proxy`, `html`, `markdown`, `screenshot`, `stream`, `cookie_jar`, `cookies`, `home_page`, `home_page_timeout`, `paginator`, `proxy`, `record`, `return_cookie`, `token_cap` |
| `scraper_type="ai"`, `agent_type="general"` | `bypass_proxy`, `html`, `markdown`, `render_javascript`, `return_cookies`, `screenshot`, `use_home_page`, `wait_for_selector` |
| `scraper_type="ai"`, `agent_type="listing"` | General AI options plus `max_pages`, `timeout`, and `stream` |
| `scraper_type="ai"`, `agent_type="map"` | `max_pages`, `max_depth`, `limit`, `include_patterns`, `exclude_patterns` |

#### `MrScraperRunExistingScraperBatch`

Runs multiple URLs with one existing AI or Manual scraper. Batch runs intentionally expose only three parameters and
do not accept per-agent advanced options.

| Parameter | Default | Description |
|---|---:|---|
| `scraper_type` | Required | `"ai"` or `"manual"`. |
| `scraper_id` | Required | Nonblank ID of the existing scraper. |
| `urls` | Required | Nonempty list of nonblank URL strings. |

```python
from pprint import pprint

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraperBatch

response = MrScraperRunExistingScraperBatch().run(
    scraper_type="manual",
    scraper_id="your-scraper-id",
    urls=[
        "https://example.com/product/1",
        "https://example.com/product/2",
    ],
)
pprint(response["result"])
```

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
