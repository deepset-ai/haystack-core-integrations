# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder, NvidiaTextEmbedder
from haystack_integrations.components.generators.nvidia import NvidiaGenerator
from haystack_integrations.components.rankers.nvidia import NvidiaRanker


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/rankings",
        "http://0.0.0.0:8888/v1/rankings",
        "http://localhost:8888/chat/completions",
        "http://localhost:8888/v1/chat/completions",
    ],
)
@pytest.mark.parametrize(
    "component",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker, NvidiaGenerator],
)
def test_base_url_invalid_not_hosted(base_url: str, component) -> None:
    with pytest.warns(UserWarning) as msg:
        component(api_url=base_url, model="x")
    assert "you may have inference and listing issues" in str(msg[0].message)


@pytest.mark.parametrize(
    "component",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker, NvidiaGenerator],
)
def test_create_without_base_url(component: type, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "valid_api_key")
    x = component()
    x.warm_up()
    assert x.api_url == "https://integrate.api.nvidia.com/v1"


@pytest.mark.parametrize(
    "component",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker, NvidiaGenerator],
)
def test_base_url_priority(component: type) -> None:
    param_url = "https://PARAM/v1"

    def get_api_url(**kwargs: Any) -> str:
        x = component(**kwargs)
        return x.api_url

    assert get_api_url(api_url=param_url) == param_url


@pytest.mark.parametrize(
    "component",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker, NvidiaGenerator],
)
@pytest.mark.parametrize(
    "api_url",
    [
        "bogus",
        "http:/",
        "http://",
        "http:/oops",
    ],
)
def test_param_api_url_negative(component: type, api_url: str) -> None:
    with pytest.raises(ValueError) as e:
        component(api_url=api_url)
    assert "Invalid api_url" in str(e.value)


@pytest.mark.parametrize(
    "component",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker, NvidiaGenerator],
)
@pytest.mark.parametrize(
    "api_url",
    ["http://nims.example.com/embedding/v1"],
)
def test_api_url_without_host(component: type, api_url: str) -> None:
    component(api_url=api_url)
