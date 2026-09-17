# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from huggingface_hub.utils import RepositoryNotFoundError
from numpy import array

from haystack_integrations.common.huggingface_api.utils import HFEmbeddingAPIType
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder


@pytest.fixture
def mock_check_valid_model():
    with (
        patch(
            "haystack_integrations.components.embedders.huggingface_api.document_embedder._check_valid_model",
            MagicMock(return_value=None),
        ) as mock,
        patch(
            "haystack_integrations.components.embedders.huggingface_api.document_embedder._check_valid_model_async",
            AsyncMock(return_value=None),
        ),
    ):
        yield mock


def mock_embedding_generation(text, **kwargs):
    return array([[random.random() for _ in range(384)] for _ in range(len(text))])


class TestInitializationAndSerialization:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self):
        model = "BAAI/bge-small-en-v1.5"
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model}
        )

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": model}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert not embedder.use_grpc
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._channel is None
        assert embedder._async_channel is None
        assert embedder._stub is None
        assert embedder._async_stub is None

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tei(self):
        url = "https://some_model.com"

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": url}
        )

        assert embedder.api_type == HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        assert embedder.api_params == {"url": url}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert not embedder.use_grpc
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._channel is None
        assert embedder._async_channel is None
        assert embedder._stub is None
        assert embedder._async_stub is None

    def test_init_tei_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tei_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_to_dict(self):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
            batch_size=128,
            progress_bar=False,
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" ",
            concurrency_limit=7,
        )

        data = embedder.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.huggingface_api.document_embedder"
            ".HuggingFaceAPIDocumentEmbedder",
            "init_parameters": {
                "api_type": "serverless_inference_api",
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
                "use_grpc": False,
                "batch_size": 128,
                "progress_bar": False,
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": " ",
                "concurrency_limit": 7,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.huggingface_api.document_embedder"
            ".HuggingFaceAPIDocumentEmbedder",
            "init_parameters": {
                "api_type": HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
                "batch_size": 128,
                "progress_bar": False,
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": " ",
                "concurrency_limit": 7,
            },
        }

        embedder = HuggingFaceAPIDocumentEmbedder.from_dict(data)

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": "BAAI/bge-small-en-v1.5"}
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.truncate
        assert embedder.normalize
        assert not embedder.use_grpc
        assert embedder.batch_size == 128
        assert not embedder.progress_bar
        assert embedder.meta_fields_to_embed == ["meta_field"]
        assert embedder.embedding_separator == " "
        assert embedder.concurrency_limit == 7


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_env_var("MISSING_HF_TOKEN"),
        )

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            embedder.warm_up()

    def test_invalid_model_is_checked_at_warm_up(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id", response=MagicMock())
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
        )
        with pytest.raises(RepositoryNotFoundError):
            embedder.warm_up()

    def test_grpc_sync_lifecycle(self):
        module = "haystack_integrations.components.embedders.huggingface_api.document_embedder"
        with (
            patch(f"{module}.InferenceClient") as mock_http_client_cls,
            patch(f"{module}.AsyncInferenceClient") as mock_async_http_client_cls,
            patch(f"{module}.grpc.insecure_channel") as mock_channel_cls,
            patch(f"{module}.grpc.aio.insecure_channel") as mock_async_channel_cls,
            patch(f"{module}.tei_pb2_grpc.EmbedStub") as mock_stub_cls,
        ):
            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
                api_params={"url": "localhost:8081"},
                use_grpc=True,
            )
            channel = mock_channel_cls.return_value
            stub = mock_stub_cls.return_value

            mock_channel_cls.assert_not_called()
            mock_async_channel_cls.assert_not_called()
            embedder.warm_up()
            embedder.warm_up()

            mock_channel_cls.assert_called_once_with("localhost:8081")
            mock_async_channel_cls.assert_not_called()
            mock_stub_cls.assert_called_once_with(channel)
            mock_http_client_cls.assert_not_called()
            mock_async_http_client_cls.assert_not_called()
            assert embedder._channel is channel
            assert embedder._stub is stub
            assert embedder._async_channel is None
            assert embedder._async_stub is None

            embedder.close()
            channel.close.assert_called_once_with()
            assert embedder._channel is None
            assert embedder._stub is None

            embedder.warm_up()
            assert mock_channel_cls.call_count == 2
            assert mock_stub_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_grpc_async_lifecycle(self):
        module = "haystack_integrations.components.embedders.huggingface_api.document_embedder"
        with (
            patch(f"{module}.InferenceClient") as mock_http_client_cls,
            patch(f"{module}.AsyncInferenceClient") as mock_async_http_client_cls,
            patch(f"{module}.grpc.insecure_channel") as mock_channel_cls,
            patch(f"{module}.grpc.aio.insecure_channel") as mock_async_channel_cls,
            patch(f"{module}.tei_pb2_grpc.EmbedStub") as mock_stub_cls,
        ):
            channel = MagicMock(close=AsyncMock())
            mock_async_channel_cls.return_value = channel
            stub = mock_stub_cls.return_value
            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
                api_params={"url": "localhost:8081"},
                use_grpc=True,
            )

            mock_async_channel_cls.assert_not_called()
            mock_channel_cls.assert_not_called()
            await embedder.warm_up_async()
            await embedder.warm_up_async()

            mock_async_channel_cls.assert_called_once_with("localhost:8081")
            mock_channel_cls.assert_not_called()
            mock_stub_cls.assert_called_once_with(channel)
            mock_http_client_cls.assert_not_called()
            mock_async_http_client_cls.assert_not_called()
            assert embedder._async_channel is channel
            assert embedder._async_stub is stub
            assert embedder._channel is None
            assert embedder._stub is None

            await embedder.close_async()
            channel.close.assert_awaited_once_with()
            assert embedder._async_channel is None
            assert embedder._async_stub is None

            await embedder.warm_up_async()
            assert mock_async_channel_cls.call_count == 2
            assert mock_stub_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.InferenceClient")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None

        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.AsyncInferenceClient")
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = MagicMock(close=AsyncMock())
        mock_client_cls.return_value = client

        await embedder.warm_up_async()
        assert embedder._async_client is client
        assert embedder._client is None

        await embedder.close_async()
        client.close.assert_awaited_once_with()
        assert embedder._async_client is None

        await embedder.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.InferenceClient")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.AsyncInferenceClient")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.AsyncInferenceClient")
    @patch("haystack_integrations.components.embedders.huggingface_api.document_embedder.InferenceClient")
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        sync_client = mock_sync_cls.return_value
        async_client = MagicMock(close=AsyncMock())
        mock_async_cls.return_value = async_client
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        embedder.warm_up()
        await embedder.warm_up_async()

        embedder.close()
        assert embedder._client is None
        assert embedder._async_client is async_client
        async_client.close.assert_not_awaited()

        await embedder.close_async()
        assert embedder._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://some_model.com"},
            token=Secret.from_token("fake-api-token"),
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://some_model.com"},
            token=Secret.from_token("fake-api-token"),
            prefix="my_prefix ",
            suffix=" my_suffix",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_embed_batch(self, mock_check_valid_model, caplog):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
            )
            embedder.warm_up()
            embeddings = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

            assert mock_embedding_patch.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

        # Check that logger warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    def test_embed_batch_wrong_embedding_shape(self, mock_check_valid_model):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        # embedding ndim != 2
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([0.1, 0.2, 0.3])

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
            )
            embedder.warm_up()

            with pytest.raises(ValueError):
                embedder._embed_batch(texts_to_embed=texts, batch_size=2)

        # embedding ndim == 2 but shape[0] != len(batch)
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
            )
            embedder.warm_up()

            with pytest.raises(ValueError):
                embedder._embed_batch(texts_to_embed=texts, batch_size=2)

    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    def test_run_on_empty_list(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            token=Secret.from_token("fake-api-token"),
        )

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    def test_run(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

            mock_embedding_patch.assert_called_once_with(
                text=[
                    "prefix Cuisine | I love cheese suffix",
                    "prefix ML | A transformer is a deep learning architecture suffix",
                ],
                truncate=None,
                normalize=None,
            )

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc, new_doc in zip(docs, documents_with_embeddings, strict=True):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert len(new_doc.embedding) == 384
            assert all(isinstance(x, float) for x in new_doc.embedding)

    def test_run_custom_batch_size(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

            assert mock_embedding_patch.call_count == 2

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    def test_embed_batch_grpc(self):
        requests = []
        stream_count = 0

        def embed_stream(batch):
            nonlocal stream_count
            stream_count += 1
            batch_requests = list(batch)
            requests.extend(batch_requests)
            return [MagicMock(embeddings=[0.1, 0.2]) for _ in batch_requests]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "localhost:8081"},
            use_grpc=True,
            progress_bar=False,
        )
        try:
            embedder.warm_up()
            assert embedder._stub is not None
            embedder._stub.EmbedStream = embed_stream
            embeddings = embedder._embed_batch_grpc(["text 1", "text 2"])
        finally:
            embedder.close()

        assert stream_count == 1
        assert embeddings == [[0.1, 0.2], [0.1, 0.2]]
        assert [(request.inputs, request.truncate, request.normalize) for request in requests] == [
            ("text 1", True, False),
            ("text 2", True, False),
        ]

    def test_adjust_api_parameters(self):
        truncate, normalize = HuggingFaceAPIDocumentEmbedder._adjust_api_parameters(
            True, False, HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        )
        assert truncate is None
        assert normalize is None

        truncate, normalize = HuggingFaceAPIDocumentEmbedder._adjust_api_parameters(
            True, False, HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        )
        assert truncate is True
        assert normalize is False

    @pytest.mark.integration
    def test_live_run_tei_grpc(self):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "localhost:8081"},
            use_grpc=True,
            progress_bar=False,
        )
        try:
            result = embedder.run([Document(content="This is a test document for embedding.")])
        finally:
            embedder.close()

        documents = result["documents"]
        assert len(documents) == 1
        assert len(documents[0].embedding) == 384
        assert all(isinstance(value, float) for value in documents[0].embedding)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei_grpc(self):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "localhost:8081"},
            use_grpc=True,
            progress_bar=False,
        )
        try:
            result = await embedder.run_async([Document(content="This is a test document for embedding.")])
        finally:
            await embedder.close_async()

        documents = result["documents"]
        assert len(documents) == 1
        assert len(documents[0].embedding) == 384
        assert all(isinstance(value, float) for value in documents[0].embedding)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_TOKEN containing the Hugging Face token to run this test.",
)
class TestIntegration:
    def test_live_run_serverless(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )
        embedder.warm_up()
        embedder._client.timeout = 10  # we want to fail fast if the server is not responding
        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.asyncio
    async def test_live_run_serverless_async(self) -> None:
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )
        await embedder.warm_up_async()
        embedder._async_client.timeout = 10  # we want to fail fast if the server is not responding
        result = await embedder.run_async(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_async_grpc(self):
        streams: list[list[tuple[str, bool, bool]]] = []
        all_streams_started = asyncio.Event()

        def embed_stream(batch):
            stream_requests: list[tuple[str, bool, bool]] = []
            streams.append(stream_requests)
            if len(streams) == 3:
                all_streams_started.set()

            async def responses():
                await all_streams_started.wait()
                async for request in batch:
                    stream_requests.append((request.inputs, request.truncate, request.normalize))
                    embedding_value = float(request.inputs.removeprefix("text "))
                    yield MagicMock(embeddings=[embedding_value])

            return responses()

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "localhost:8081"},
            use_grpc=True,
            concurrency_limit=3,
            progress_bar=False,
        )
        try:
            await embedder.warm_up_async()
            assert embedder._async_stub is not None
            embedder._async_stub.EmbedStream = embed_stream
            embeddings = await embedder._embed_batch_grpc_async(
                ["text 1", "text 2", "text 3", "text 4", "text 5", "text 6", "text 7"]
            )
        finally:
            await embedder.close_async()

        assert embeddings == [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
        assert streams == [
            [("text 1", True, False), ("text 2", True, False)],
            [("text 3", True, False), ("text 4", True, False)],
            [("text 5", True, False), ("text 6", True, False), ("text 7", True, False)],
        ]


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_embed_batch_async(self, mock_check_valid_model, caplog):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                concurrency_limit=4,
            )
            await embedder.warm_up_async()
            embeddings = await embedder._embed_batch_async(texts_to_embed=texts, batch_size=2)

            assert mock_embedding_patch.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

        # Check that logger warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    @pytest.mark.asyncio
    async def test_embed_batch_async_wrong_embedding_shape(self, mock_check_valid_model):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        # embedding ndim != 2
        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([0.1, 0.2, 0.3])

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                concurrency_limit=1,
            )
            await embedder.warm_up_async()

            with pytest.raises(ValueError):
                await embedder._embed_batch_async(texts_to_embed=texts, batch_size=2)

        # embedding ndim == 2 but shape[0] != len(batch)
        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                concurrency_limit=1,
            )
            await embedder.warm_up_async()

            with pytest.raises(ValueError):
                await embedder._embed_batch_async(texts_to_embed=texts, batch_size=2)

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError):
            await embedder.run_async(text=list_integers_input)

    @pytest.mark.asyncio
    async def test_run_async_on_empty_list(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            token=Secret.from_token("fake-api-token"),
        )

        empty_list_input = []
        result = await embedder.run_async(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list.

    @pytest.mark.asyncio
    async def test_run_async(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = await embedder.run_async(documents=docs)

            mock_embedding_patch.assert_called_once_with(
                text=[
                    "prefix Cuisine | I love cheese suffix",
                    "prefix ML | A transformer is a deep learning architecture suffix",
                ],
                truncate=None,
                normalize=None,
            )

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.asyncio
    async def test_run_async_custom_batch_size(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = await embedder.run_async(documents=docs)

            assert mock_embedding_patch.call_count == 2

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)
