# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from haystack import Document

from haystack_integrations.components.websearch.ddgs import DDGSWebSearch

DDGS_PATH = "haystack_integrations.components.websearch.ddgs.ddgs_websearch.DDGS"

SAMPLE_RESULTS = [
    {"title": "Haystack", "href": "https://haystack.deepset.ai", "body": "Haystack is an open-source AI framework."},
    {"title": "deepset", "href": "https://www.deepset.ai", "body": "deepset builds enterprise LLM tooling."},
]


def test_init_default():
    ws = DDGSWebSearch()
    assert ws.top_k == 10
    assert ws.backend == "auto"
    assert ws.region == "us-en"
    assert ws.safesearch == "moderate"
    assert ws.search_params == {}


def test_init_custom():
    ws = DDGSWebSearch(
        top_k=3, backend="duckduckgo, brave", region="de-de", safesearch="off", search_params={"page": 2}
    )
    assert ws.top_k == 3
    assert ws.backend == "duckduckgo, brave"
    assert ws.region == "de-de"
    assert ws.safesearch == "off"
    assert ws.search_params == {"page": 2}


def test_to_dict():
    ws = DDGSWebSearch(top_k=5, region="de-de")
    assert ws.to_dict() == {
        "type": "haystack_integrations.components.websearch.ddgs.ddgs_websearch.DDGSWebSearch",
        "init_parameters": {
            "top_k": 5,
            "backend": "auto",
            "region": "de-de",
            "safesearch": "moderate",
            "search_params": {},
        },
    }


def test_from_dict():
    ws = DDGSWebSearch.from_dict(
        {
            "type": "haystack_integrations.components.websearch.ddgs.ddgs_websearch.DDGSWebSearch",
            "init_parameters": {
                "top_k": 7,
                "backend": "auto",
                "region": "us-en",
                "safesearch": "on",
                "search_params": {},
            },
        }
    )
    assert ws.top_k == 7
    assert ws.safesearch == "on"


def test_to_and_from_dict_roundtrip():
    ws = DDGSWebSearch(top_k=4, backend="google", search_params={"timelimit": "w"})
    restored = DDGSWebSearch.from_dict(ws.to_dict())
    assert restored.to_dict() == ws.to_dict()


@patch(DDGS_PATH)
def test_run(mock_ddgs):
    mock_ddgs.return_value.text.return_value = SAMPLE_RESULTS
    ws = DDGSWebSearch(top_k=2)
    result = ws.run(query="what is haystack?")

    assert [d.content for d in result["documents"]] == [r["body"] for r in SAMPLE_RESULTS]
    assert result["documents"][0].meta == {"title": "Haystack", "url": "https://haystack.deepset.ai"}
    assert result["links"] == ["https://haystack.deepset.ai", "https://www.deepset.ai"]

    _, kwargs = mock_ddgs.return_value.text.call_args
    assert kwargs["max_results"] == 2
    assert kwargs["backend"] == "auto"
    assert kwargs["region"] == "us-en"
    assert kwargs["safesearch"] == "moderate"


@patch(DDGS_PATH)
def test_run_collects_links_only_when_url_present(mock_ddgs):
    mock_ddgs.return_value.text.return_value = [
        {"title": "no url", "body": "x"},
        {"title": "ok", "href": "https://ok.example", "body": "y"},
    ]
    result = DDGSWebSearch().run(query="q")
    assert len(result["documents"]) == 2
    assert result["links"] == ["https://ok.example"]


@patch(DDGS_PATH)
def test_search_params_override_config(mock_ddgs):
    mock_ddgs.return_value.text.return_value = []
    ws = DDGSWebSearch(region="us-en", search_params={"region": "fr-fr", "timelimit": "d"})
    ws.run(query="q")
    _, kwargs = mock_ddgs.return_value.text.call_args
    assert kwargs["region"] == "fr-fr"
    assert kwargs["timelimit"] == "d"


@pytest.mark.asyncio
@patch(DDGS_PATH)
async def test_run_async(mock_ddgs):
    mock_ddgs.return_value.text.return_value = SAMPLE_RESULTS
    result = await DDGSWebSearch(top_k=2).run_async(query="what is haystack?")
    assert len(result["documents"]) == 2
    assert result["links"] == ["https://haystack.deepset.ai", "https://www.deepset.ai"]


@pytest.mark.integration
def test_web_search_integration():
    result = DDGSWebSearch(top_k=3).run(query="What is the deepset Haystack framework?")
    assert len(result["documents"]) > 0
    assert all(isinstance(doc, Document) for doc in result["documents"])
    assert all(doc.meta.get("url") for doc in result["documents"])
    assert result["links"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_web_search_async_integration():
    result = await DDGSWebSearch(top_k=3).run_async(query="What is the deepset Haystack framework?")
    assert len(result["documents"]) > 0
    assert result["links"]
