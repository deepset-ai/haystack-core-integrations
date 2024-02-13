import pytest
from requests import HTTPError

from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

class TestOllamaDocumentEmbedder:
    def test_init_defaults(self):
        embedder = OllamaDocumentEmbedder()

        assert embedder.timeout == 120
        assert embedder.generation_kwargs == {}
        assert embedder.url == "http://localhost:11434/api/embeddings"
        assert embedder.model == "orca-mini"

    def test_init(self):
        embedder = OllamaDocumentEmbedder(
            model="llama2",
            url="http://my-custom-endpoint:11434/api/embeddings",
            generation_kwargs={"temperature": 0.5},
            timeout=3000,
        )

        assert embedder.timeout == 3000
        assert embedder.generation_kwargs == {"temperature": 0.5}
        assert embedder.url == "http://my-custom-endpoint:11434/api/embeddings"
        assert embedder.model == "llama2"

    @pytest.mark.integration
    def test_model_not_found(self):
        embedder = OllamaDocumentEmbedder(model="cheese")

        with pytest.raises(HTTPError):
            embedder.run("hello")

    @pytest.mark.integration
    def import_text_in_embedder(self):
        embedder = OllamaDocumentEmbedder(model="orca-mini")
        
        embedder.run("This is a text string. This should not work.")
        with pytest.raises(TypeError):
            embedder.run("hello")

    @pytest.mark.integration
    def test_run(self):
        embedder = OllamaDocumentEmbedder(model="orca-mini")
        list_of_docs = [Document(content="This is a document containing some text.")]
        reply = embedder.run(list_of_docs)

        assert isinstance(reply, dict)
        assert all(isinstance(element, float) for element in reply["embedding"])
        assert reply["meta"]["model"] == "orca-mini"
