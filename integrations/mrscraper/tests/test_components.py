# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

import pytest
from haystack import Pipeline
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

from .conftest import FAKE_TOKEN

COMPONENT_CLASSES = [
    MrScraperGetAccountInfo,
    MrScraperCrawlWebsiteUrls,
    MrScraperSearchGoogleSerp,
    MrScraperExtractPageByPrompt,
    MrScraperExtractListings,
    MrScraperExtractStructuredData,
    MrScraperFetchRenderedHtml,
    MrScraperGetResults,
    MrScraperGetLatestResults,
    MrScraperGetResultDetail,
    MrScraperCreatePromptScraper,
    MrScraperCreateListingScraper,
    MrScraperCreateWebsiteCrawlScraper,
    MrScraperRunExistingScraper,
    MrScraperRunExistingScraperBatch,
]


@pytest.mark.parametrize("component_class", COMPONENT_CLASSES)
def test_all_components_registered_with_one_result_socket_and_async(component_class):
    instance = component_class()
    assert list(instance.__haystack_output__._sockets_dict) == ["result"]
    assert instance.__haystack_supports_async__ is True
    assert inspect.getdoc(component_class)
    assert inspect.getdoc(component_class.run)
    assert inspect.signature(component_class.run).parameters


@pytest.mark.parametrize("component_class", COMPONENT_CLASSES)
def test_component_serialization_roundtrip(component_class):
    component = component_class(api_key=Secret.from_env_var("MRSCRAPER_API_TOKEN"), connect_timeout=2, read_timeout=9)
    data = component.to_dict()
    serialized = json.dumps(data)
    assert FAKE_TOKEN not in serialized
    assert data["init_parameters"]["api_key"] == {
        "type": "env_var",
        "env_vars": ["MRSCRAPER_API_TOKEN"],
        "strict": True,
    }
    restored = component_class.from_dict(data)
    assert restored.api_key == component.api_key
    assert restored.connect_timeout == 2
    assert restored.read_timeout == 9


def test_token_secret_is_runtime_only_and_not_serializable():
    component = MrScraperGetAccountInfo(api_key=Secret.from_token(FAKE_TOKEN))
    with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
        component.to_dict()


def test_pipeline_serialization_roundtrip_without_resolving_secret(monkeypatch):
    monkeypatch.delenv("MRSCRAPER_API_TOKEN", raising=False)
    pipeline = Pipeline()
    pipeline.add_component("search", MrScraperSearchGoogleSerp())
    data = pipeline.to_dict()
    assert FAKE_TOKEN not in json.dumps(data)
    restored = Pipeline.from_dict(data)
    assert isinstance(restored.get_component("search"), MrScraperSearchGoogleSerp)


def test_pipeline_invokes_component_with_native_output(http_recorder):
    pipeline = Pipeline()
    pipeline.add_component("search", MrScraperSearchGoogleSerp(api_key=Secret.from_token(FAKE_TOKEN)))
    result = pipeline.run({"search": {"query": "Haystack"}})
    assert result == {"search": {"result": {"ok": True}}}
    assert len(http_recorder.requests) == 1


def test_constructor_does_not_resolve_missing_secret(monkeypatch):
    monkeypatch.delenv("MRSCRAPER_API_TOKEN", raising=False)
    component = MrScraperGetAccountInfo()
    assert component.api_key == Secret.from_env_var("MRSCRAPER_API_TOKEN")


def test_missing_secret_fails_before_transport(monkeypatch, http_recorder):
    monkeypatch.delenv("MRSCRAPER_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MRSCRAPER_API_TOKEN"):
        MrScraperGetAccountInfo().run()
    assert not http_recorder.requests


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_timeout_validation(value):
    with pytest.raises(ValueError, match="connect_timeout"):
        MrScraperGetAccountInfo(connect_timeout=value)
