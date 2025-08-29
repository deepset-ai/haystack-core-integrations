import pytest
from haystack import Document
from ollama._types import ResponseError

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder


class TestOllamaDocumentEmbedder:
    def test_init_defaults(self):
        embedder = OllamaDocumentEmbedder()

        assert embedder.keep_alive is None
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
            keep_alive="10m",
        )

        assert embedder.keep_alive == "10m"
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
        embedder = OllamaDocumentEmbedder(model="all-minilm")

        with pytest.raises(TypeError):
            embedder.run("This is a text string. This should not work.")

    @pytest.mark.integration
    def test_run(self):
        embedder = OllamaDocumentEmbedder(model="all-minilm", batch_size=2)
        list_of_docs = [
            Document(content="Llamas are amazing animals known for their soft wool and gentle demeanor."),
            Document(content="The Andes mountains are the natural habitat of many llamas."),
            Document(content="Llamas have been used as pack animals for centuries, especially in South America."),
        ]
        result = embedder.run(list_of_docs)

        assert result["meta"]["model"] == "all-minilm"
        documents = result["documents"]
        assert len(documents) == 3
        assert all(isinstance(element, float) for document in documents for element in document.embedding)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = OllamaDocumentEmbedder(model="all-minilm", batch_size=2)
        list_of_docs = [
            Document(content="Llamas are amazing animals known for their soft wool and gentle demeanor."),
            Document(content="The Andes mountains are the natural habitat of many llamas."),
            Document(content="Llamas have been used as pack animals for centuries, especially in South America."),
        ]
        result = await embedder.run_async(list_of_docs)

        assert result["meta"]["model"] == "all-minilm"
        documents = result["documents"]
        assert len(documents) == 3
        assert all(isinstance(element, float) for document in documents for element in document.embedding)
