import pytest
from ollama._types import ResponseError

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


class TestOllamaTextEmbedder:
    def test_init_defaults(self):
        embedder = OllamaTextEmbedder()

        assert embedder.timeout == 120
        assert embedder.generation_kwargs == {}
        assert embedder.url == "http://localhost:11434"
        assert embedder.model == "nomic-embed-text"

    def test_init(self):
        embedder = OllamaTextEmbedder(
            model="llama2",
            url="http://my-custom-endpoint:11434",
            generation_kwargs={"temperature": 0.5},
            timeout=3000,
        )

        assert embedder.timeout == 3000
        assert embedder.generation_kwargs == {"temperature": 0.5}
        assert embedder.url == "http://my-custom-endpoint:11434"
        assert embedder.model == "llama2"

    @pytest.mark.integration
    def test_model_not_found(self):
        embedder = OllamaTextEmbedder(model="cheese")

        with pytest.raises(ResponseError):
            embedder.run("hello")

    @pytest.mark.integration
    def test_run(self):
        embedder = OllamaTextEmbedder(model="nomic-embed-text")

        reply = embedder.run("hello")

        assert isinstance(reply, dict)
        assert all(isinstance(element, float) for element in reply["embedding"])
        assert reply["meta"]["model"] == "nomic-embed-text"
