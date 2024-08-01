import pytest
from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder, NvidiaTextEmbedder
from haystack_integrations.components.generators.nvidia import NvidiaGenerator


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
    "embedder",
    [NvidiaDocumentEmbedder, NvidiaTextEmbedder],
)
def test_base_url_invalid_not_hosted(base_url: str, embedder) -> None:
    with pytest.raises(ValueError):
        embedder(api_url=base_url, model="x")


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
