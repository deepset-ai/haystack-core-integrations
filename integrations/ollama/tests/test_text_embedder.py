import pytest
from ollama._types import ResponseError

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


class TestOllamaTextEmbedder:
    def test_init_defaults(self):
        embedder = OllamaTextEmbedder()

        assert embedder.keep_alive is None
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
            keep_alive="10m",
        )

        assert embedder.keep_alive == "10m"
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
        embedder = OllamaTextEmbedder(model="all-minilm")

        text = "hello"
        reply = embedder.run(text=text)

        assert isinstance(reply, dict)
        assert all(isinstance(element, float) for element in reply["embedding"])
        assert reply["meta"]["model"] == "all-minilm"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = OllamaTextEmbedder(model="all-minilm")

        text = "hello"
        reply = await embedder.run_async(text=text)

        assert isinstance(reply, dict)
        assert all(isinstance(element, float) for element in reply["embedding"])
        assert reply["meta"]["model"] == "all-minilm"
