from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Document
from haystack.core.serialization import default_from_dict, default_to_dict
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

    def test_dimensions_default_is_none(self):
        embedder = OllamaDocumentEmbedder()
        assert embedder.dimensions is None

    def test_dimensions_stored_on_instance(self):
        embedder = OllamaDocumentEmbedder(dimensions=512)
        assert embedder.dimensions == 512

    def test_dimensions_passed_to_embed_client(self):
        embedder = OllamaDocumentEmbedder(dimensions=512)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        embedder._client.embed = MagicMock(return_value=mock_response)

        embedder._embed_batch(["hello world"], batch_size=32)

        call_kwargs = embedder._client.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 512

    def test_none_dimensions_passed_to_embed_client(self):
        embedder = OllamaDocumentEmbedder(dimensions=None)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        embedder._client.embed = MagicMock(return_value=mock_response)

        embedder._embed_batch(["hello"], batch_size=32)

        call_kwargs = embedder._client.embed.call_args.kwargs
        assert call_kwargs["dimensions"] is None

    @pytest.mark.asyncio
    async def test_dimensions_passed_to_async_embed_client(self):
        embedder = OllamaDocumentEmbedder(dimensions=256)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        await embedder._embed_batch_async(["hello"], batch_size=32)

        call_kwargs = embedder._async_client.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 256

    def test_to_dict_contains_dimensions(self):
        embedder = OllamaDocumentEmbedder(dimensions=512)
        embedder_dict = default_to_dict(
            embedder,
            model=embedder.model,
            url=embedder.url,
            generation_kwargs=embedder.generation_kwargs,
            timeout=embedder.timeout,
            keep_alive=embedder.keep_alive,
            prefix=embedder.prefix,
            suffix=embedder.suffix,
            progress_bar=embedder.progress_bar,
            meta_fields_to_embed=embedder.meta_fields_to_embed,
            embedding_separator=embedder.embedding_separator,
            batch_size=embedder.batch_size,
            dimensions=embedder.dimensions,
        )
        assert embedder_dict["init_parameters"]["dimensions"] == 512

    def test_to_dict_contains_dimensions_none(self):
        embedder = OllamaDocumentEmbedder()
        embedder_dict = default_to_dict(
            embedder,
            model=embedder.model,
            url=embedder.url,
            generation_kwargs=embedder.generation_kwargs,
            timeout=embedder.timeout,
            keep_alive=embedder.keep_alive,
            prefix=embedder.prefix,
            suffix=embedder.suffix,
            progress_bar=embedder.progress_bar,
            meta_fields_to_embed=embedder.meta_fields_to_embed,
            embedding_separator=embedder.embedding_separator,
            batch_size=embedder.batch_size,
            dimensions=embedder.dimensions,
        )
        assert embedder_dict["init_parameters"]["dimensions"] is None

    def test_from_dict_restores_dimensions(self):
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.ollama.document_embedder.OllamaDocumentEmbedder",
            "init_parameters": {
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
                "dimensions": 512,
            },
        }
        embedder = default_from_dict(OllamaDocumentEmbedder, embedder_dict)
        assert embedder.dimensions == 512
