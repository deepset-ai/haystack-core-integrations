# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

from haystack_integrations.components.mrscraper import MrScraperRunExistingScraper, MrScraperRunExistingScraperBatch

from .conftest import FAKE_TOKEN


def scraper():
    return MrScraperRunExistingScraper(api_key=Secret.from_token(FAKE_TOKEN))


def batch_scraper():
    return MrScraperRunExistingScraperBatch(api_key=Secret.from_token(FAKE_TOKEN))


def test_general_ai_omits_advanced_options_and_uses_ai_endpoint(http_recorder):
    scraper().run(scraper_type="ai", scraper_id="s1", url="https://example.com")
    request = http_recorder.requests[0]
    assert request.url.path == "/api/v1/scrapers-ai-rerun"
    assert request.body == {
        "scraperId": "s1",
        "url": "https://example.com",
        "maxRetry": 3,
    }


def test_listing_ai_only_sends_explicit_advanced_options(http_recorder):
    scraper().run(
        scraper_type="ai",
        scraper_id="s1",
        url="https://example.com",
        agent_type="listing",
        stream=False,
        wait_for_selector=" ",
    )
    body = http_recorder.requests[0].body
    assert body["stream"] is False
    assert "maxPages" not in body
    assert "timeout" not in body
    assert "waitForSelector" not in body


def test_map_ai_only_sends_explicit_options_and_zero_depth(http_recorder):
    scraper().run(
        scraper_type="ai",
        scraper_id="s1",
        url="https://example.com",
        agent_type="map",
        max_depth=0,
        include_patterns="^https://example.com",
    )
    assert http_recorder.requests[0].body == {
        "scraperId": "s1",
        "url": "https://example.com",
        "maxRetry": 3,
        "maxDepth": 0,
        "includePatterns": "^https://example.com",
    }


def test_manual_only_sends_explicit_options_and_preserves_json_values(http_recorder):
    scraper().run(
        scraper_type="manual",
        scraper_id="s1",
        url="https://example.com",
        cookies=[],
        paginator={},
        screenshot=False,
        token_cap=0,
    )
    request = http_recorder.requests[0]
    assert request.url.path == "/api/v1/scrapers-manual-rerun"
    assert request.body == {
        "scraperId": "s1",
        "url": "https://example.com",
        "maxRetry": 3,
        "cookies": [],
        "paginator": {},
        "screenshot": "false",
        "tokenCap": 0,
    }


def test_manual_screenshot_true_is_lowercase_string(http_recorder):
    scraper().run(scraper_type="manual", scraper_id="s1", url="https://example.com", screenshot=True)
    assert http_recorder.requests[0].body["screenshot"] == "true"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scraper_type": "manual", "agent_type": "general"}, "AI-only"),
        ({"scraper_type": "manual", "render_javascript": False}, "AI-only"),
        ({"scraper_type": "ai", "agent_type": "general", "cookies": []}, "manual-only"),
        ({"scraper_type": "ai", "agent_type": "general", "max_pages": 1}, "General AI"),
        ({"scraper_type": "ai", "agent_type": "listing", "max_depth": 0}, "Listing AI"),
        ({"scraper_type": "ai", "agent_type": "map", "html": False}, "Map AI"),
    ],
)
def test_incompatible_options_rejected_before_transport(kwargs, match, http_recorder):
    with pytest.raises(ValueError, match=match):
        scraper().run(scraper_id="s1", url="https://example.com", **kwargs)
    assert not http_recorder.requests


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retry": True},
        {"max_retry": 1.5},
        {"scraper_type": "ai", "agent_type": "listing", "max_pages": 0},
        {"scraper_type": "manual", "home_page_timeout": 0},
        {"scraper_type": "manual", "cookies": {}},
        {"scraper_type": "manual", "paginator": []},
    ],
)
def test_run_validation(kwargs, http_recorder):
    base = {"scraper_type": "ai", "scraper_id": "s1", "url": "https://example.com"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        scraper().run(**base)
    assert not http_recorder.requests


@pytest.mark.parametrize(
    ("scraper_type", "path"),
    [
        ("ai", "/api/v1/scrapers-ai-rerun/bulk"),
        ("manual", "/api/v1/scrapers-manual-rerun/bulk"),
    ],
)
def test_batch_endpoints_and_native_url_array(scraper_type, path, http_recorder):
    urls = [" https://example.com/1 ", "https://example.com/2"]
    batch_scraper().run(scraper_type=scraper_type, scraper_id="s1", urls=urls)
    request = http_recorder.requests[0]
    assert request.url.path == path
    assert request.body == {
        "scraperId": "s1",
        "urls": ["https://example.com/1", "https://example.com/2"],
    }


@pytest.mark.parametrize("urls", [[], [""], ["https://example.com", "  "], "https://example.com"])
def test_batch_rejects_invalid_urls(urls, http_recorder):
    with pytest.raises(ValueError, match="urls"):
        batch_scraper().run(scraper_type="ai", scraper_id="s1", urls=urls)
    assert not http_recorder.requests
