import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from haystack.core.serialization import default_from_dict, default_to_dict
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
        assert embedder._client is None
        assert embedder._async_client is None

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

    def test_dimensions_default_is_none(self):
        embedder = OllamaTextEmbedder()
        assert embedder.dimensions is None

    def test_dimensions_stored_on_instance(self):
        embedder = OllamaTextEmbedder(dimensions=256)
        assert embedder.dimensions == 256

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.Client")
    def test_dimensions_passed_to_embed_client(self, mock_client_cls):
        embedder = OllamaTextEmbedder(dimensions=256)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed.return_value = mock_response

        embedder.run(text="hello world")

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 256

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.Client")
    def test_none_dimensions_passed_to_embed_client(self, mock_client_cls):
        embedder = OllamaTextEmbedder(dimensions=None)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed.return_value = mock_response

        embedder.run(text="hello")

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] is None

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.AsyncClient")
    def test_dimensions_passed_to_async_embed_client(self, mock_client_cls):
        embedder = OllamaTextEmbedder(dimensions=128)
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value.embed = AsyncMock(return_value=mock_response)

        asyncio.run(embedder.run_async(text="hello"))

        call_kwargs = mock_client_cls.return_value.embed.call_args.kwargs
        assert call_kwargs["dimensions"] == 128

    def test_to_dict_contains_dimensions(self):

        embedder = OllamaTextEmbedder(dimensions=256)
        embedder_dict = default_to_dict(
            embedder,
            model=embedder.model,
            url=embedder.url,
            generation_kwargs=embedder.generation_kwargs,
            timeout=embedder.timeout,
            keep_alive=embedder.keep_alive,
            dimensions=embedder.dimensions,
        )
        assert embedder_dict["init_parameters"]["dimensions"] == 256

    def test_to_dict_contains_dimensions_none(self):

        embedder = OllamaTextEmbedder()
        embedder_dict = default_to_dict(
            embedder,
            model=embedder.model,
            url=embedder.url,
            generation_kwargs=embedder.generation_kwargs,
            timeout=embedder.timeout,
            keep_alive=embedder.keep_alive,
            dimensions=embedder.dimensions,
        )
        assert embedder_dict["init_parameters"]["dimensions"] is None

    def test_from_dict_restores_dimensions(self):

        embedder_dict = {
            "type": "haystack_integrations.components.embedders.ollama.text_embedder.OllamaTextEmbedder",
            "init_parameters": {
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
                "dimensions": 256,
            },
        }
        embedder = default_from_dict(OllamaTextEmbedder, embedder_dict)
        assert embedder.dimensions == 256


class TestComponentLifecycle:
    @patch("haystack_integrations.components.embedders.ollama.text_embedder.Client")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = OllamaTextEmbedder()
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None

        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.AsyncClient")
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = OllamaTextEmbedder()
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

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.Client")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = OllamaTextEmbedder()
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once_with(host=embedder.url, timeout=embedder.timeout)

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.AsyncClient")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = OllamaTextEmbedder()
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once_with(host=embedder.url, timeout=embedder.timeout)

    async def test_close_is_safe_without_warm_up(self):
        embedder = OllamaTextEmbedder()
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

    @patch("haystack_integrations.components.embedders.ollama.text_embedder.AsyncClient")
    @patch("haystack_integrations.components.embedders.ollama.text_embedder.Client")
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        embedder = OllamaTextEmbedder()
        mock_async_cls.return_value.close = AsyncMock()
        embedder.warm_up()
        await embedder.warm_up_async()

        embedder.close()
        mock_sync_cls.return_value.close.assert_called_once_with()
        assert embedder._client is None
        assert embedder._async_client is mock_async_cls.return_value

        await embedder.close_async()
        assert embedder._async_client is None
