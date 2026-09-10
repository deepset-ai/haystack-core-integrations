from unittest.mock import AsyncMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import default_from_dict, default_to_dict
from ollama._types import ResponseError

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder


class TestInitialization:
    def test_init_defaults(self):
        embedder = OllamaDocumentEmbedder()

        assert embedder.keep_alive is None
        assert embedder.timeout == 120
        assert embedder.generation_kwargs == {}
        assert embedder.url == "http://localhost:11434"
        assert embedder.model == "nomic-embed-text"
        assert embedder._client is None
        assert embedder._async_client is None

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

    def test_dimensions_default_is_none(self):
        embedder = OllamaDocumentEmbedder()
        assert embedder.dimensions is None

    def test_dimensions_stored_on_instance(self):
        embedder = OllamaDocumentEmbedder(dimensions=512)
        assert embedder.dimensions == 512


class TestSerialization:
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


class TestComponentLifecycle:
    @patch("haystack_integrations.components.embedders.ollama.document_embedder.Client")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder()
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None

        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.AsyncClient")
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder()
        client = mock_client_cls.return_value
        client.close = AsyncMock()

        await embedder.warm_up_async()
        assert embedder._async_client is client
        assert embedder._client is None

        await embedder.close_async()
        client.close.assert_awaited_once_with()
        assert embedder._async_client is None

        await embedder.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.Client")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder()
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once_with(host=embedder.url, timeout=embedder.timeout)

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.AsyncClient")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder()
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once_with(host=embedder.url, timeout=embedder.timeout)

    async def test_close_is_safe_without_warm_up(self):
        embedder = OllamaDocumentEmbedder()
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.AsyncClient")
    @patch("haystack_integrations.components.embedders.ollama.document_embedder.Client")
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        embedder = OllamaDocumentEmbedder()
        mock_async_cls.return_value.close = AsyncMock()
        embedder.warm_up()
        await embedder.warm_up_async()

        embedder.close()
        mock_sync_cls.return_value.close.assert_called_once_with()
        assert embedder._client is None
        assert embedder._async_client is mock_async_cls.return_value

        await embedder.close_async()
        assert embedder._async_client is None


class TestRun:
    def test_rejects_string_input(self):
        embedder = OllamaDocumentEmbedder(model="all-minilm")

        with pytest.raises(TypeError):
            embedder.run("This is a text string. This should not work.")

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.Client")
    def test_dimensions_passed_to_embed_client(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder(dimensions=512)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed.return_value = mock_response
        embedder.warm_up()

        embedder._embed_batch(["hello world"], batch_size=32)

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 512

    @patch("haystack_integrations.components.embedders.ollama.document_embedder.Client")
    def test_none_dimensions_passed_to_embed_client(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder(dimensions=None)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed.return_value = mock_response
        embedder.warm_up()

        embedder._embed_batch(["hello"], batch_size=32)

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.ollama.document_embedder.AsyncClient")
    async def test_dimensions_passed_to_async_embed_client(self, mock_client_cls):
        embedder = OllamaDocumentEmbedder(dimensions=256)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed = AsyncMock(return_value=mock_response)
        await embedder.warm_up_async()

        await embedder._embed_batch_async(["hello"], batch_size=32)

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 256


@pytest.mark.integration
class TestIntegration:
    def test_model_not_found(self):
        embedder = OllamaDocumentEmbedder(model="cheese")

        with pytest.raises(ResponseError):
            embedder.run([Document("hello")])

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
