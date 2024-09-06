import pytest
from haystack import Document
from ollama._types import ResponseError

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder


class TestOllamaDocumentEmbedder:
    def test_init_defaults(self):
        embedder = OllamaDocumentEmbedder()

        assert embedder.timeout == 120
        assert embedder.generation_kwargs == {}
        assert embedder.url == "http://localhost:11434"
        assert embedder.model == "nomic-embed-text"

    def test_init(self):
        embedder = OllamaDocumentEmbedder(
            model="nomic-embed-text",
            url="http://my-custom-endpoint:11434",
            generation_kwargs={"temperature": 0.5},
            timeout=3000,
        )

        assert embedder.timeout == 3000
        assert embedder.generation_kwargs == {"temperature": 0.5}
        assert embedder.url == "http://my-custom-endpoint:11434"
        assert embedder.model == "nomic-embed-text"

    @pytest.mark.integration
    def test_model_not_found(self):
        embedder = OllamaDocumentEmbedder(model="cheese")

        with pytest.raises(ResponseError):
            embedder.run([Document("hello")])

    @pytest.mark.integration
    def import_text_in_embedder(self):
        embedder = OllamaDocumentEmbedder(model="nomic-embed-text")

        with pytest.raises(TypeError):
            embedder.run("This is a text string. This should not work.")

    @pytest.mark.integration
    def test_run(self):
        embedder = OllamaDocumentEmbedder(model="nomic-embed-text")
        list_of_docs = [Document(content="This is a document containing some text.")]
        reply = embedder.run(list_of_docs)

        assert isinstance(reply, dict)
        assert all(isinstance(element, float) for element in reply["documents"][0].embedding)
        assert reply["meta"]["model"] == "nomic-embed-text"
