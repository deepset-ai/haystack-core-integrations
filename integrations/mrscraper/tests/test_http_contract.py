# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from urllib.parse import parse_qs

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.components.mrscraper import (
    MrScraperCrawlWebsiteUrls,
    MrScraperCreateListingScraper,
    MrScraperCreatePromptScraper,
    MrScraperCreateWebsiteCrawlScraper,
    MrScraperExtractListings,
    MrScraperExtractPageByPrompt,
    MrScraperExtractStructuredData,
    MrScraperFetchRenderedHtml,
    MrScraperGetAccountInfo,
    MrScraperGetLatestResults,
    MrScraperGetResultDetail,
    MrScraperGetResults,
    MrScraperRunExistingScraper,
    MrScraperRunExistingScraperBatch,
    MrScraperSearchGoogleSerp,
)
from haystack_integrations.utils.mrscraper.client import MrScraperError
from haystack_integrations.utils.mrscraper.presets import load_structured_data_prompts

from .conftest import FAKE_TOKEN


def component(component_class):
    return component_class(api_key=Secret.from_token(FAKE_TOKEN), connect_timeout=3, read_timeout=11)


def test_account_contract_and_finite_timeouts(http_recorder):
    result = component(MrScraperGetAccountInfo).run()
    request = http_recorder.requests[0]
    assert result == {"result": {"ok": True}}
    assert request.method == "GET"
    assert str(request.url) == "https://api.app.mrscraper.com/api/v1/subscription-accounts"
    assert request.headers["x-api-token"] == FAKE_TOKEN
    assert request.timeout.connect == 3
    assert request.timeout.read == 11


def test_map_payload_uses_camel_case_and_omits_blanks(http_recorder):
    component(MrScraperCrawlWebsiteUrls).run(
        url="https://example.com",
        max_depth=0,
        max_pages=1,
        limit=1,
        include_patterns=" ",
        exclude_patterns="/private",
    )
    assert http_recorder.requests[0].body == {
        "graph": "map",
        "url": "https://example.com",
        "maxDepth": 0,
        "maxPages": 1,
        "limit": 1,
        "excludePatterns": "/private",
    }


@pytest.mark.parametrize("bad_value", [True, 1.2, 0, -1])
def test_integer_validation_rejects_bool_fraction_and_below_minimum(bad_value, http_recorder):
    with pytest.raises(ValueError, match="limit"):
        component(MrScraperCrawlWebsiteUrls).run(url="https://example.com", limit=bad_value)
    assert not http_recorder.requests


def test_serp_json_contract_and_bearer_auth(http_recorder):
    http_recorder.response_json = [1, {"title": "result"}, False]
    result = component(MrScraperSearchGoogleSerp).run(
        query="Haystack",
        region="id",
        language="en",
        page=2,
        format="json",
        render_js=False,
    )
    request = http_recorder.requests[0]
    assert result["result"] == [1, {"title": "result"}, False]
    assert str(request.url) == "https://sync.scraper.mrscraper.com/api/google/serp/v2/sync"
    assert request.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "x-api-token" not in request.headers
    assert request.body == {
        "query": "Haystack",
        "region": "id",
        "language": "en",
        "page": 2,
        "format": "json",
        "renderJs": False,
    }


def test_serp_html_is_exact_text(http_recorder):
    html = "<!doctype html>\n<html>  exact </html>\n"
    http_recorder.response_text = html
    http_recorder.response_headers = {"content-type": "text/html; charset=utf-8"}
    result = component(MrScraperSearchGoogleSerp).run(query="test", format="html")
    assert result == {"result": html}


def test_general_schema_is_compact_and_appended_once(http_recorder):
    schema = {"café": "string", "price": 0}
    component(MrScraperExtractPageByPrompt).run(
        url="https://example.com/item",
        prompt="Extract it",
        output_schema=schema,
        mode="Cheap",
        proxy_country="ID",
    )
    message = 'Extract it\n\nReturn the output as JSON matching this schema:\n{"café":"string","price":0}'
    assert http_recorder.requests[0].body == {
        "graph": "general",
        "url": "https://example.com/item",
        "message": message,
        "mode": "Cheap",
        "proxyCountry": "ID",
    }


def test_general_schema_append_is_idempotent(http_recorder):
    block = 'Return the output as JSON matching this schema:\n{"name":"string"}'
    prompt = f"Extract it\n\n{block}"
    component(MrScraperExtractPageByPrompt).run(
        url="https://example.com/item", prompt=prompt, output_schema={"name": "string"}
    )
    assert http_recorder.requests[0].body["message"] == prompt


@pytest.mark.parametrize("upstream", [{"a": 1}, [1, 2], False, 0, "json string", None])
def test_native_json_objects_arrays_and_scalars_are_preserved(upstream, http_recorder):
    if upstream is None:
        http_recorder.response_text = "null"
        http_recorder.response_headers = {"content-type": "application/json"}
    else:
        http_recorder.response_json = upstream
    assert component(MrScraperGetAccountInfo).run()["result"] == upstream


def test_listing_schema_uses_distinct_label_and_preserves_empty_dict(http_recorder):
    component(MrScraperExtractListings).run(url="https://example.com/items", output_schema={}, max_pages=1)
    assert http_recorder.requests[0].body["message"] == "Return each item as JSON matching this schema:\n{}"


def test_structured_presets_are_exact_bundled_source_and_used_for_every_category():
    path = "src/haystack_integrations/utils/mrscraper/structured_data_prompts.json"
    with open(path, "rb") as preset_file:
        preset_bytes = preset_file.read()
    expected_hash = "3d9c15e8ebe7ad8cb04281251311200c1d3413452f14f252dc9ed3a8aae8533a"
    assert hashlib.sha256(preset_bytes).hexdigest() == expected_hash
    parsed = json.loads(preset_bytes)
    assert parsed == load_structured_data_prompts()
    for category, prompt in parsed.items():
        payload = MrScraperExtractStructuredData._payload("https://example.com", category, "Super", None)
        assert payload["message"] == prompt
        assert "category" not in payload


def test_rendered_query_body_split_and_lowercase_booleans(http_recorder):
    component(MrScraperFetchRenderedHtml).run(
        url="https://example.com/app",
        max_retries=0,
        timeout=1,
        screenshot=True,
        screenshot_mode="top",
        html=False,
        markdown=True,
        token_cap=1,
        wait_for_selector="#ready",
        wait_until="domcontentloaded",
        block_resources=False,
        home_page=False,
        return_cookie=False,
        super_mode=False,
    )
    request = http_recorder.requests[0]
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.mrscraper.com"
    query = parse_qs(request.url.query.decode())
    assert query == {
        "token": [FAKE_TOKEN],
        "browserRendering": ["true"],
        "timeout": ["1"],
        "geoCode": ["us"],
        "html": ["false"],
        "markdown": ["true"],
        "screenshot": ["top"],
        "proxyCountry": ["us"],
        "waitUntil": ["domcontentloaded"],
        "waitForSelector": ["#ready"],
    }
    assert request.body == {"url": "https://example.com/app", "maxRetries": 0, "tokenCap": 1}


def test_rendered_omits_screenshot_mode_when_disabled(http_recorder):
    component(MrScraperFetchRenderedHtml).run(url="https://example.com", screenshot=False, screenshot_mode="top")
    assert "screenshot" not in parse_qs(http_recorder.requests[0].url.query.decode())


def test_rendered_omits_unused_advanced_options(http_recorder):
    component(MrScraperFetchRenderedHtml).run(
        url="https://example.com",
        screenshot=None,
        block_resources=None,
        home_page=None,
        return_cookie=None,
        super_mode=None,
    )
    request = http_recorder.requests[0]
    query = parse_qs(request.url.query.decode())
    assert not {
        "screenshot",
        "waitForSelector",
        "waitUntil",
        "blockResources",
        "returnCookie",
        "super",
    }.intersection(query)
    assert request.body == {"url": "https://example.com", "maxRetries": 3}


def test_rendered_sends_enabled_advanced_options(http_recorder):
    component(MrScraperFetchRenderedHtml).run(
        url="https://example.com",
        block_resources=True,
        home_page=True,
        return_cookie=True,
        super_mode=True,
    )
    request = http_recorder.requests[0]
    query = parse_qs(request.url.query.decode())
    assert query["blockResources"] == ["true"]
    assert query["returnCookie"] == ["true"]
    assert query["super"] == ["true"]
    assert request.body["homePage"] is True


def test_rendered_requires_mode_only_when_screenshot_is_enabled(http_recorder):
    with pytest.raises(ValueError, match="screenshot_mode"):
        component(MrScraperFetchRenderedHtml).run(url="https://example.com", screenshot=True)
    assert not http_recorder.requests


def test_results_filters_sorting_latest_and_encoded_detail(http_recorder):
    component(MrScraperGetResults).run(scraper_id="scraper / one", page=2, page_size=0 + 1, sort_order="ASC")
    component(MrScraperGetLatestResults).run(scraper_id="scraper / one", count=3)
    component(MrScraperGetResultDetail).run(result_id="result/with space?#")
    first_query = parse_qs(http_recorder.requests[0].url.query.decode())
    assert first_query == {
        "filters[scraperId]": ["scraper / one"],
        "page": ["2"],
        "pageSize": ["1"],
        "sort": ["createdAt"],
        "sortOrder": ["ASC"],
    }
    second_query = parse_qs(http_recorder.requests[1].url.query.decode())
    assert second_query["page"] == ["1"]
    assert second_query["pageSize"] == ["3"]
    assert str(http_recorder.requests[2].url).endswith("/api/v1/results/result%2Fwith%20space%3F%23")


@pytest.mark.asyncio
async def test_sync_async_request_parity(http_recorder):
    scraper = component(MrScraperExtractPageByPrompt)
    sync_result = scraper.run(url="https://example.com", prompt="extract")
    async_result = await scraper.run_async(url="https://example.com", prompt="extract")
    assert sync_result == async_result
    assert http_recorder.requests[0].method == http_recorder.requests[1].method
    assert http_recorder.requests[0].url == http_recorder.requests[1].url
    assert http_recorder.requests[0].body == http_recorder.requests[1].body


@pytest.mark.asyncio
async def test_all_operation_families_support_true_async(http_recorder):
    await component(MrScraperGetAccountInfo).run_async()
    await component(MrScraperCrawlWebsiteUrls).run_async(url="https://example.com")
    await component(MrScraperSearchGoogleSerp).run_async(query="test")
    await component(MrScraperExtractListings).run_async(url="https://example.com")
    await component(MrScraperExtractStructuredData).run_async(url="https://example.com")
    await component(MrScraperFetchRenderedHtml).run_async(url="https://example.com")
    await component(MrScraperGetResults).run_async(scraper_id="s1")
    await component(MrScraperGetLatestResults).run_async(scraper_id="s1")
    await component(MrScraperGetResultDetail).run_async(result_id="r1")
    await component(MrScraperCreatePromptScraper).run_async(url="https://example.com")
    await component(MrScraperCreateListingScraper).run_async(url="https://example.com")
    await component(MrScraperCreateWebsiteCrawlScraper).run_async(url="https://example.com")
    await component(MrScraperRunExistingScraper).run_async(
        scraper_type="manual", scraper_id="s1", url="https://example.com"
    )
    await component(MrScraperRunExistingScraperBatch).run_async(
        scraper_type="ai", scraper_id="s1", urls=["https://example.com"]
    )
    assert len(http_recorder.requests) == 14


def test_http_error_detail_is_truncated_and_token_redacted(http_recorder):
    http_recorder.response_status = 400
    http_recorder.response_text = f"bad token={FAKE_TOKEN} " + "x" * 1000
    with pytest.raises(MrScraperError) as error:
        component(MrScraperFetchRenderedHtml).run(url="https://example.com")
    message = str(error.value)
    assert "status 400" in message
    assert FAKE_TOKEN not in message
    assert "[REDACTED]" in message
    assert len(message) < 600


def test_transport_error_redacts_token(monkeypatch):
    def fail(*_args, **_kwargs):
        request = httpx.Request("POST", f"https://api.mrscraper.com/?token={FAKE_TOKEN}")
        message = f"could not connect to {request.url}"
        raise httpx.ConnectError(message, request=request)

    monkeypatch.setattr(httpx.Client, "request", fail)
    with pytest.raises(MrScraperError) as error:
        component(MrScraperFetchRenderedHtml).run(url="https://example.com")
    assert FAKE_TOKEN not in str(error.value)
