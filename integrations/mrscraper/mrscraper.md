---
title: "MrScraper"
id: integrations-mrscraper
description: "MrScraper integration for Haystack"
slug: "/integrations-mrscraper"
---


## haystack_integrations.components.mrscraper.account

### MrScraperGetAccountInfo

Bases: <code>MrScraperComponent</code>

Retrieve MrScraper account details, token usage, and token limits.

#### run

```python
run() -> dict[str, Any]
```

Get account information.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async() -> dict[str, Any]
```

Asynchronously get account information.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

## haystack_integrations.components.mrscraper.discovery

### MrScraperCrawlWebsiteUrls

Bases: <code>MrScraperComponent</code>

Discover URLs by crawling links from one starting website immediately.

#### run

```python
run(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    limit: int = 50,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
) -> dict[str, Any]
```

Crawl a website for URLs.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL to crawl.
- **max_depth** (<code>int</code>) – Maximum link depth to crawl.
- **max_pages** (<code>int</code>) – Maximum number of pages to evaluate.
- **limit** (<code>int</code>) – Maximum number of discovered URLs to return. Must be at least 1.
- **include_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to include.
- **exclude_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to exclude.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    limit: int = 50,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
) -> dict[str, Any]
```

Asynchronously crawl a website for URLs.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL to crawl.
- **max_depth** (<code>int</code>) – Maximum link depth to crawl.
- **max_pages** (<code>int</code>) – Maximum number of pages to evaluate.
- **limit** (<code>int</code>) – Maximum number of discovered URLs to return. Must be at least 1.
- **include_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to include.
- **exclude_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to exclude.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperSearchGoogleSerp

Bases: <code>MrScraperComponent</code>

Search Google synchronously and return native JSON results or exact HTML text.

#### run

```python
run(
    query: str,
    region: str = "us",
    language: str = "en",
    page: int = 1,
    format: Literal["json", "html"] = "json",
    render_js: bool = False,
) -> dict[str, Any]
```

Search Google through the MrScraper SERP v2 API.

**Parameters:**

- **query** (<code>str</code>) – Nonblank Google search query.
- **region** (<code>str</code>) – Two-letter country or region code.
- **language** (<code>str</code>) – Two-letter result language code.
- **page** (<code>int</code>) – Results page number. Must be at least 1.
- **format** (<code>Literal['json', 'html']</code>) – Response format: `json` or `html`.
- **render_js** (<code>bool</code>) – Whether to render JavaScript before collecting results.

**Returns:**

- <code>dict\[str, Any\]</code> – Native decoded JSON for JSON format, or exact response text for HTML format, under `result`.

#### run_async

```python
run_async(
    query: str,
    region: str = "us",
    language: str = "en",
    page: int = 1,
    format: Literal["json", "html"] = "json",
    render_js: bool = False,
) -> dict[str, Any]
```

Asynchronously search Google through the MrScraper SERP v2 API.

**Parameters:**

- **query** (<code>str</code>) – Nonblank Google search query.
- **region** (<code>str</code>) – Two-letter country or region code.
- **language** (<code>str</code>) – Two-letter result language code.
- **page** (<code>int</code>) – Results page number. Must be at least 1.
- **format** (<code>Literal['json', 'html']</code>) – Response format: `json` or `html`.
- **render_js** (<code>bool</code>) – Whether to render JavaScript before collecting results.

**Returns:**

- <code>dict\[str, Any\]</code> – Native decoded JSON for JSON format, or exact response text for HTML format, under `result`.

## haystack_integrations.components.mrscraper.extraction

### MrScraperExtractPageByPrompt

Bases: <code>MrScraperComponent</code>

Immediately extract data from one page using a natural-language prompt and optional JSON schema.

#### run

```python
run(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Extract one page with a General AI scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL to extract.
- **prompt** (<code>str | None</code>) – Optional extraction instructions.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON shape appended compactly to the prompt.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Asynchronously extract one page with a General AI scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL to extract.
- **prompt** (<code>str | None</code>) – Optional extraction instructions.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON shape appended compactly to the prompt.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperExtractListings

Bases: <code>MrScraperComponent</code>

Immediately extract repeated listings or paginated content with an optional item schema.

#### run

```python
run(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_pages: int = 1,
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Extract listings or paginated content.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL to extract.
- **prompt** (<code>str | None</code>) – Optional instructions describing each listing item.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON item shape appended compactly to the prompt.
- **max_pages** (<code>int</code>) – Maximum pagination pages to scrape. Must be at least 1.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_pages: int = 1,
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Asynchronously extract listings or paginated content.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL to extract.
- **prompt** (<code>str | None</code>) – Optional instructions describing each listing item.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON item shape appended compactly to the prompt.
- **max_pages** (<code>int</code>) – Maximum pagination pages to scrape. Must be at least 1.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperExtractStructuredData

Bases: <code>MrScraperComponent</code>

Extract structured data using one of ten exact bundled MrScraper category prompts.

#### run

```python
run(
    url: str,
    category: StructuredCategory = "article",
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Extract a supported structured-data category.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL to extract.
- **category** (<code>StructuredCategory</code>) – Structured prompt preset category.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    category: StructuredCategory = "article",
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Asynchronously extract a supported structured-data category.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL to extract.
- **category** (<code>StructuredCategory</code>) – Structured prompt preset category.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperFetchRenderedHtml

Bases: <code>MrScraperComponent</code>

Fetch a JavaScript-rendered page with browser controls and native JSON or exact text output.

#### run

```python
run(
    url: str,
    max_retries: int = 3,
    timeout: int = 300,
    geo_code: str = "us",
    proxy_country: str = "us",
    screenshot: bool | None = False,
    screenshot_mode: Literal["full", "top"] | None = None,
    html: bool = True,
    markdown: bool = False,
    token_cap: int | None = None,
    wait_for_selector: str | None = None,
    wait_until: (
        Literal["domcontentloaded", "load", "networkidle"] | None
    ) = None,
    block_resources: bool | None = False,
    home_page: bool | None = False,
    return_cookie: bool | None = False,
    super_mode: bool | None = False,
) -> dict[str, Any]
```

Fetch a rendered page through the MrScraper stealth browser.

**Parameters:**

- **url** (<code>str</code>) – Nonblank target URL, sent only as request data.
- **max_retries** (<code>int</code>) – Maximum upstream retry attempts. Must be at least 0.
- **timeout** (<code>int</code>) – Page-load timeout in seconds. Must be at least 1.
- **geo_code** (<code>str</code>) – Two-letter geolocation code.
- **proxy_country** (<code>str</code>) – Two-letter proxy country code.
- **screenshot** (<code>bool | None</code>) – Whether to capture a screenshot.
- **screenshot_mode** (<code>Literal['full', 'top'] | None</code>) – Optional screenshot coverage. Required only when screenshot is enabled.
- **html** (<code>bool</code>) – Whether the response should include rendered HTML.
- **markdown** (<code>bool</code>) – Whether the response should include converted Markdown.
- **token_cap** (<code>int | None</code>) – Optional maximum processing token allowance. Must be at least 1 when provided.
- **wait_for_selector** (<code>str | None</code>) – Optional nonblank CSS selector to await.
- **wait_until** (<code>Literal['domcontentloaded', 'load', 'networkidle'] | None</code>) – Optional browser lifecycle event to await.
- **block_resources** (<code>bool | None</code>) – Enable blocking images, fonts, and stylesheets. Omitted when false.
- **home_page** (<code>bool | None</code>) – Enable visiting the site home page first. Omitted when false.
- **return_cookie** (<code>bool | None</code>) – Enable returning browser cookies. Omitted when false.
- **super_mode** (<code>bool | None</code>) – Enable a real device for stronger scraping capabilities. Omitted when false.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing native decoded JSON or exact text under `result`.

#### run_async

```python
run_async(
    url: str,
    max_retries: int = 3,
    timeout: int = 300,
    geo_code: str = "us",
    proxy_country: str = "us",
    screenshot: bool | None = False,
    screenshot_mode: Literal["full", "top"] | None = None,
    html: bool = True,
    markdown: bool = False,
    token_cap: int | None = None,
    wait_for_selector: str | None = None,
    wait_until: (
        Literal["domcontentloaded", "load", "networkidle"] | None
    ) = None,
    block_resources: bool | None = False,
    home_page: bool | None = False,
    return_cookie: bool | None = False,
    super_mode: bool | None = False,
) -> dict[str, Any]
```

Asynchronously fetch a rendered page through the MrScraper stealth browser.

**Parameters:**

- **url** (<code>str</code>) – Nonblank target URL, sent only as request data.
- **max_retries** (<code>int</code>) – Maximum upstream retry attempts. Must be at least 0.
- **timeout** (<code>int</code>) – Page-load timeout in seconds. Must be at least 1.
- **geo_code** (<code>str</code>) – Two-letter geolocation code.
- **proxy_country** (<code>str</code>) – Two-letter proxy country code.
- **screenshot** (<code>bool | None</code>) – Whether to capture a screenshot.
- **screenshot_mode** (<code>Literal['full', 'top'] | None</code>) – Optional screenshot coverage. Required only when screenshot is enabled.
- **html** (<code>bool</code>) – Whether the response should include rendered HTML.
- **markdown** (<code>bool</code>) – Whether the response should include converted Markdown.
- **token_cap** (<code>int | None</code>) – Optional maximum processing token allowance. Must be at least 1 when provided.
- **wait_for_selector** (<code>str | None</code>) – Optional nonblank CSS selector to await.
- **wait_until** (<code>Literal['domcontentloaded', 'load', 'networkidle'] | None</code>) – Optional browser lifecycle event to await.
- **block_resources** (<code>bool | None</code>) – Enable blocking images, fonts, and stylesheets. Omitted when false.
- **home_page** (<code>bool | None</code>) – Enable visiting the site home page first. Omitted when false.
- **return_cookie** (<code>bool | None</code>) – Enable returning browser cookies. Omitted when false.
- **super_mode** (<code>bool | None</code>) – Enable a real device for stronger scraping capabilities. Omitted when false.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing native decoded JSON or exact text under `result`.

## haystack_integrations.components.mrscraper.results

### MrScraperGetResults

Bases: <code>MrScraperComponent</code>

Get one explicitly paginated and sorted page of results for an existing scraper.

#### run

```python
run(
    scraper_id: str,
    page: int = 1,
    page_size: int = 10,
    sort_by: Literal["createdAt"] = "createdAt",
    sort_order: Literal["ASC", "DESC"] = "DESC",
) -> dict[str, Any]
```

Get paginated scraper results.

**Parameters:**

- **scraper_id** (<code>str</code>) – Nonblank ID of the scraper whose results should be fetched.
- **page** (<code>int</code>) – Page number.
- **page_size** (<code>int</code>) – Results per page.
- **sort_by** (<code>Literal['createdAt']</code>) – Sort field; currently only `createdAt`.
- **sort_order** (<code>Literal['ASC', 'DESC']</code>) – Sort direction, `ASC` or `DESC`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    scraper_id: str,
    page: int = 1,
    page_size: int = 10,
    sort_by: Literal["createdAt"] = "createdAt",
    sort_order: Literal["ASC", "DESC"] = "DESC",
) -> dict[str, Any]
```

Asynchronously get paginated scraper results.

**Parameters:**

- **scraper_id** (<code>str</code>) – Nonblank ID of the scraper whose results should be fetched.
- **page** (<code>int</code>) – Page number.
- **page_size** (<code>int</code>) – Results per page.
- **sort_by** (<code>Literal['createdAt']</code>) – Sort field; currently only `createdAt`.
- **sort_order** (<code>Literal['ASC', 'DESC']</code>) – Sort direction, `ASC` or `DESC`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperGetLatestResults

Bases: <code>MrScraperComponent</code>

Get only the newest N results for an existing scraper.

#### run

```python
run(scraper_id: str, count: int = 10) -> dict[str, Any]
```

Get the latest scraper results.

**Parameters:**

- **scraper_id** (<code>str</code>) – Nonblank ID of the scraper whose latest results should be fetched.
- **count** (<code>int</code>) – Number of newest results to return.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(scraper_id: str, count: int = 10) -> dict[str, Any]
```

Asynchronously get the latest scraper results.

**Parameters:**

- **scraper_id** (<code>str</code>) – Nonblank ID of the scraper whose latest results should be fetched.
- **count** (<code>int</code>) – Number of newest results to return.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperGetResultDetail

Bases: <code>MrScraperComponent</code>

Get one complete scraper result by its result ID.

#### run

```python
run(result_id: str) -> dict[str, Any]
```

Get one result in detail.

**Parameters:**

- **result_id** (<code>str</code>) – Nonblank result ID, URL-encoded as one path segment.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(result_id: str) -> dict[str, Any]
```

Asynchronously get one result in detail.

**Parameters:**

- **result_id** (<code>str</code>) – Nonblank result ID, URL-encoded as one path segment.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

## haystack_integrations.components.mrscraper.scraper_creation

### MrScraperCreatePromptScraper

Bases: <code>MrScraperComponent</code>

Create a reusable General AI scraper from a prompt instead of describing an immediate-only task.

#### run

```python
run(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Create a reusable prompt-based scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL used to create the scraper.
- **prompt** (<code>str | None</code>) – Optional extraction instructions.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON shape appended compactly to the prompt.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    mode: Mode = "Super",
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Asynchronously create a reusable prompt-based scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank page URL used to create the scraper.
- **prompt** (<code>str | None</code>) – Optional extraction instructions.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON shape appended compactly to the prompt.
- **mode** (<code>Mode</code>) – Scraping mode, `Super` or `Cheap`.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperCreateListingScraper

Bases: <code>MrScraperComponent</code>

Create a reusable Listing AI scraper for repeated or paginated items.

#### run

```python
run(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_pages: int = 1,
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Create a reusable listing scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL used to create the scraper.
- **prompt** (<code>str | None</code>) – Optional instructions describing each listing item.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON item shape appended compactly to the prompt.
- **max_pages** (<code>int</code>) – Maximum pagination pages to scrape. Must be at least 1.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_pages: int = 1,
    proxy_country: str | None = None,
) -> dict[str, Any]
```

Asynchronously create a reusable listing scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL used to create the scraper.
- **prompt** (<code>str | None</code>) – Optional instructions describing each listing item.
- **output_schema** (<code>dict\[str, Any\] | None</code>) – Optional JSON item shape appended compactly to the prompt.
- **max_pages** (<code>int</code>) – Maximum pagination pages to scrape. Must be at least 1.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperCreateWebsiteCrawlScraper

Bases: <code>MrScraperComponent</code>

Create a reusable Map AI scraper for future website URL discovery runs.

#### run

```python
run(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    limit: int = 50,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
) -> dict[str, Any]
```

Create a reusable website-crawl scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL used to create the scraper.
- **max_depth** (<code>int</code>) – Maximum link depth to crawl.
- **max_pages** (<code>int</code>) – Maximum pages to evaluate.
- **limit** (<code>int</code>) – Maximum discovered URLs to return. Must be at least 1.
- **include_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to include.
- **exclude_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to exclude.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    limit: int = 50,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
) -> dict[str, Any]
```

Asynchronously create a reusable website-crawl scraper.

**Parameters:**

- **url** (<code>str</code>) – Nonblank starting URL used to create the scraper.
- **max_depth** (<code>int</code>) – Maximum link depth to crawl.
- **max_pages** (<code>int</code>) – Maximum pages to evaluate.
- **limit** (<code>int</code>) – Maximum discovered URLs to return. Must be at least 1.
- **include_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to include.
- **exclude_patterns** (<code>str | None</code>) – Optional pipe-separated regular expressions for URLs to exclude.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

## haystack_integrations.components.mrscraper.scraper_runs

### conditional_run_tool_schema

```python
conditional_run_tool_schema(generated_schema: dict[str, Any]) -> dict[str, Any]
```

Turn the flat component schema into Manual and agent-specific AI branches for Agent tools.

### MrScraperRunExistingScraper

Bases: <code>MrScraperComponent</code>

Run one URL with an existing AI or manual scraper using strict type-specific options.

#### run

```python
run(
    scraper_type: ScraperType,
    scraper_id: str,
    url: str,
    max_retry: int = 3,
    proxy_country: str | None = None,
    agent_type: AgentType | None = None,
    max_pages: int | None = None,
    timeout: int | None = None,
    bypass_proxy: bool | None = None,
    html: bool | None = None,
    markdown: bool | None = None,
    render_javascript: bool | None = None,
    return_cookies: bool | None = None,
    screenshot: bool | None = None,
    stream: bool | None = None,
    use_home_page: bool | None = None,
    wait_for_selector: str | None = None,
    max_depth: int | None = None,
    limit: int | None = None,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
    cookie_jar: str | None = None,
    cookies: list[dict[str, Any]] | None = None,
    home_page: bool | None = None,
    home_page_timeout: int | None = None,
    paginator: dict[str, Any] | None = None,
    proxy: str | None = None,
    record: bool | None = None,
    return_cookie: bool | None = None,
    token_cap: int | None = None,
) -> dict[str, Any]
```

Run one URL with an existing scraper.

`None` on an advanced setting means it is omitted so the API can apply its own default.

**Parameters:**

- **scraper_type** (<code>ScraperType</code>) – Existing scraper type, `ai` or `manual`.
- **scraper_id** (<code>str</code>) – Nonblank ID of the existing scraper.
- **url** (<code>str</code>) – Nonblank URL to process.
- **max_retry** (<code>int</code>) – Maximum retry attempts. Must be at least 0.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.
- **agent_type** (<code>AgentType | None</code>) – AI agent type; defaults to `general` for direct component use and must be absent for Manual.
- **max_pages** (<code>int | None</code>) – Optional Listing or Map page limit; valid only for those AI agent types.
- **timeout** (<code>int | None</code>) – Optional Listing or Manual timeout; invalid for General and Map AI.
- **bypass_proxy** (<code>bool | None</code>) – Optional General/Listing AI or Manual proxy-bypass setting.
- **html** (<code>bool | None</code>) – Whether to include HTML; valid for General/Listing AI and Manual.
- **markdown** (<code>bool | None</code>) – Whether to include Markdown; valid for General/Listing AI and Manual.
- **render_javascript** (<code>bool | None</code>) – Whether General/Listing AI should render JavaScript.
- **return_cookies** (<code>bool | None</code>) – Whether General/Listing AI should return cookies.
- **screenshot** (<code>bool | None</code>) – Whether to capture a screenshot; Manual sends lowercase string booleans.
- **stream** (<code>bool | None</code>) – Whether to stream results; valid for Listing AI and Manual.
- **use_home_page** (<code>bool | None</code>) – Whether General/Listing AI should first visit the home page.
- **wait_for_selector** (<code>str | None</code>) – Optional CSS selector for General/Listing AI.
- **max_depth** (<code>int | None</code>) – Optional Map AI crawl depth with minimum 0.
- **limit** (<code>int | None</code>) – Optional Map AI result limit with minimum 1.
- **include_patterns** (<code>str | None</code>) – Optional Map AI pipe-separated include patterns.
- **exclude_patterns** (<code>str | None</code>) – Optional Map AI pipe-separated exclude patterns.
- **cookie_jar** (<code>str | None</code>) – Optional Manual cookie-jar identifier or serialized value.
- **cookies** (<code>list\[dict\[str, Any\]\] | None</code>) – Optional Manual browser cookies as a list of dictionaries.
- **home_page** (<code>bool | None</code>) – Whether Manual should first visit the home page.
- **home_page_timeout** (<code>int | None</code>) – Optional Manual home-page timeout with minimum 1.
- **paginator** (<code>dict\[str, Any\] | None</code>) – Optional Manual pagination configuration dictionary.
- **proxy** (<code>str | None</code>) – Optional Manual proxy URL.
- **record** (<code>bool | None</code>) – Whether to record the Manual browser session.
- **return_cookie** (<code>bool | None</code>) – Whether Manual should return browser cookies.
- **token_cap** (<code>int | None</code>) – Optional Manual result token cap with minimum 0.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    scraper_type: ScraperType,
    scraper_id: str,
    url: str,
    max_retry: int = 3,
    proxy_country: str | None = None,
    agent_type: AgentType | None = None,
    max_pages: int | None = None,
    timeout: int | None = None,
    bypass_proxy: bool | None = None,
    html: bool | None = None,
    markdown: bool | None = None,
    render_javascript: bool | None = None,
    return_cookies: bool | None = None,
    screenshot: bool | None = None,
    stream: bool | None = None,
    use_home_page: bool | None = None,
    wait_for_selector: str | None = None,
    max_depth: int | None = None,
    limit: int | None = None,
    include_patterns: str | None = None,
    exclude_patterns: str | None = None,
    cookie_jar: str | None = None,
    cookies: list[dict[str, Any]] | None = None,
    home_page: bool | None = None,
    home_page_timeout: int | None = None,
    paginator: dict[str, Any] | None = None,
    proxy: str | None = None,
    record: bool | None = None,
    return_cookie: bool | None = None,
    token_cap: int | None = None,
) -> dict[str, Any]
```

Asynchronously run one URL with an existing scraper.

`None` on an advanced setting means it is omitted so the API can apply its own default.

**Parameters:**

- **scraper_type** (<code>ScraperType</code>) – Existing scraper type, `ai` or `manual`.
- **scraper_id** (<code>str</code>) – Nonblank ID of the existing scraper.
- **url** (<code>str</code>) – Nonblank URL to process.
- **max_retry** (<code>int</code>) – Maximum retry attempts. Must be at least 0.
- **proxy_country** (<code>str | None</code>) – Optional two-letter proxy country code.
- **agent_type** (<code>AgentType | None</code>) – AI agent type; defaults to `general` for direct component use and must be absent for Manual.
- **max_pages** (<code>int | None</code>) – Optional Listing or Map page limit; valid only for those AI agent types.
- **timeout** (<code>int | None</code>) – Optional Listing or Manual timeout; invalid for General and Map AI.
- **bypass_proxy** (<code>bool | None</code>) – Optional General/Listing AI or Manual proxy-bypass setting.
- **html** (<code>bool | None</code>) – Whether to include HTML; valid for General/Listing AI and Manual.
- **markdown** (<code>bool | None</code>) – Whether to include Markdown; valid for General/Listing AI and Manual.
- **render_javascript** (<code>bool | None</code>) – Whether General/Listing AI should render JavaScript.
- **return_cookies** (<code>bool | None</code>) – Whether General/Listing AI should return cookies.
- **screenshot** (<code>bool | None</code>) – Whether to capture a screenshot; Manual sends lowercase string booleans.
- **stream** (<code>bool | None</code>) – Whether to stream results; valid for Listing AI and Manual.
- **use_home_page** (<code>bool | None</code>) – Whether General/Listing AI should first visit the home page.
- **wait_for_selector** (<code>str | None</code>) – Optional CSS selector for General/Listing AI.
- **max_depth** (<code>int | None</code>) – Optional Map AI crawl depth with minimum 0.
- **limit** (<code>int | None</code>) – Optional Map AI result limit with minimum 1.
- **include_patterns** (<code>str | None</code>) – Optional Map AI pipe-separated include patterns.
- **exclude_patterns** (<code>str | None</code>) – Optional Map AI pipe-separated exclude patterns.
- **cookie_jar** (<code>str | None</code>) – Optional Manual cookie-jar identifier or serialized value.
- **cookies** (<code>list\[dict\[str, Any\]\] | None</code>) – Optional Manual browser cookies as a list of dictionaries.
- **home_page** (<code>bool | None</code>) – Whether Manual should first visit the home page.
- **home_page_timeout** (<code>int | None</code>) – Optional Manual home-page timeout with minimum 1.
- **paginator** (<code>dict\[str, Any\] | None</code>) – Optional Manual pagination configuration dictionary.
- **proxy** (<code>str | None</code>) – Optional Manual proxy URL.
- **record** (<code>bool | None</code>) – Whether to record the Manual browser session.
- **return_cookie** (<code>bool | None</code>) – Whether Manual should return browser cookies.
- **token_cap** (<code>int | None</code>) – Optional Manual result token cap with minimum 0.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

### MrScraperRunExistingScraperBatch

Bases: <code>MrScraperComponent</code>

Run multiple URLs in one batch with an existing AI or manual scraper.

#### run

```python
run(
    scraper_type: ScraperType, scraper_id: str, urls: list[str]
) -> dict[str, Any]
```

Run an existing scraper for a batch of URLs.

**Parameters:**

- **scraper_type** (<code>ScraperType</code>) – Existing scraper type, `ai` or `manual`.
- **scraper_id** (<code>str</code>) – Nonblank ID of the existing scraper.
- **urls** (<code>list\[str\]</code>) – Nonempty list of nonblank URL strings.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

#### run_async

```python
run_async(
    scraper_type: ScraperType, scraper_id: str, urls: list[str]
) -> dict[str, Any]
```

Asynchronously run an existing scraper for a batch of URLs.

**Parameters:**

- **scraper_type** (<code>ScraperType</code>) – Existing scraper type, `ai` or `manual`.
- **scraper_id** (<code>str</code>) – Nonblank ID of the existing scraper.
- **urls** (<code>list\[str\]</code>) – Nonempty list of nonblank URL strings.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the unmodified decoded upstream value under `result`.

## haystack_integrations.tools.mrscraper.toolset

### MrScraperToolset

Bases: <code>Toolset</code>

A serializable set of 15 independently named MrScraper ComponentTools.

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("MRSCRAPER_API_TOKEN"),
    groups: list[MrScraperToolGroup] | None = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 300.0,
) -> None
```

Create the MrScraper tools.

**Parameters:**

- **api_key** (<code>Secret</code>) – MrScraper API token. Defaults to the `MRSCRAPER_API_TOKEN` environment variable.
- **groups** (<code>list\[MrScraperToolGroup\] | None</code>) – Optional subset of `account`, `discovery`, `extraction`, `results`, `scraper_creation`,
  and `scraper_runs`. All groups are included by default.
- **connect_timeout** (<code>float</code>) – Maximum seconds to establish an HTTP connection.
- **read_timeout** (<code>float</code>) – Maximum seconds to wait while reading an HTTP response.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize configuration without resolving the API token.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MrScraperToolset
```

Deserialize the toolset and rebuild its independently configured components.
