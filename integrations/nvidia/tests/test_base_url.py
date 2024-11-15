# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder, NvidiaRanker],
)
def test_base_url_invalid_not_hosted(base_url: str, component) -> None:
    with pytest.raises(ValueError):
        component(api_url=base_url, model="x")


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080/v1/embeddings", "http://0.0.0.0:8888/v1/embeddings"],
)
@pytest.mark.parametrize(
    "embedder",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder],
)
def test_base_url_valid_embedder(base_url: str, embedder) -> None:
    with pytest.warns(UserWarning):
        embedder(api_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8080/v1/chat/completions",
        "http://0.0.0.0:8888/v1/chat/completions",
    ],
)
def test_base_url_valid_generator(base_url: str) -> None:
    with pytest.warns(UserWarning):
        NvidiaGenerator(
            api_url=base_url,
            model="mistralai/mixtral-8x7b-instruct-v0.1",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/rankings",
        "http://0.0.0.0:8888/v1/rankings",
        "http://localhost:8888/chat/completions",
    ],
)
def test_base_url_invalid_generator(base_url: str) -> None:
    with pytest.raises(ValueError):
        NvidiaGenerator(api_url=base_url, model="x")


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080/v1/ranking", "http://0.0.0.0:8888/v1/ranking"],
)
def test_base_url_valid_ranker(base_url: str) -> None:
    with pytest.warns(UserWarning):
        NvidiaRanker(api_url=base_url)
