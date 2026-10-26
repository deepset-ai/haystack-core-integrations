# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret
from openai import APIError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.vllm import VLLMSparseDocumentEmbedder

MODEL = "prithivida/Splade_PP_en_v1"
API_BASE_URL = "http://localhost:8003/v1"


def _fake_response(embeddings: list[list[float]], prompt_tokens: int = 1, total_tokens: int = 1):
    return CreateEmbeddingResponse(
        object="list",
        model="fake-model",
        data=[Embedding(object="embedding", index=i, embedding=e) for i, e in enumerate(embeddings)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def _api_error() -> APIError:
    return APIError(message="boom", request=MagicMock(), body=None)


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.model == MODEL
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None
        assert embedder.raise_on_failure is False
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_with_parameters(self):
        embedder = VLLMSparseDocumentEmbedder(
            model=MODEL,
            api_key=Secret.from_token("test-api-key"),
            api_base_url="http://my-vllm-server:8000/v1",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
            raise_on_failure=True,
            extra_parameters={"truncate_prompt_tokens": 256},
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "http://my-vllm-server:8000/v1"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"
        assert embedder.timeout == 10.0
        assert embedder.max_retries == 2
        assert embedder.http_client_kwargs == {"proxy": "https://proxy.example.com"}
        assert embedder.raise_on_failure is True
        assert embedder.extra_parameters == {"truncate_prompt_tokens": 256}

    def test_to_dict(self):
        component_dict = component_to_dict(VLLMSparseDocumentEmbedder(model=MODEL), "embedder")
        assert component_dict == {
            "type": (
                "haystack_integrations.components.embedders.vllm.sparse_document_embedder.VLLMSparseDocumentEmbedder"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "model": MODEL,
                "api_base_url": "http://localhost:8000/v1",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
                "raise_on_failure": False,
                "extra_parameters": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.embedders.vllm.sparse_document_embedder.VLLMSparseDocumentEmbedder"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "model": MODEL,
                "api_base_url": "http://localhost:8000/v1",
                "prefix": "",
                "suffix": "",
                "batch_size": 16,
                "progress_bar": False,
                "meta_fields_to_embed": ["title"],
                "embedding_separator": "\n",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
                "raise_on_failure": False,
                "extra_parameters": None,
            },
        }
        embedder = component_from_dict(VLLMSparseDocumentEmbedder, data, "embedder")
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.model == MODEL
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 16
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["title"]
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None
        assert embedder.raise_on_failure is False
        assert embedder.extra_parameters is None


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            embedder.warm_up()

    @patch("haystack_integrations.components.embedders.vllm.sparse_document_embedder._create_openai_client")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None
        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.vllm.sparse_document_embedder._create_async_openai_client")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
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

    @patch("haystack_integrations.components.embedders.vllm.sparse_document_embedder._create_openai_client")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once()

    @patch("haystack_integrations.components.embedders.vllm.sparse_document_embedder._create_async_openai_client")
    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None


class TestRun:
    def test_prepare_texts_to_embed(self):
        embedder = VLLMSparseDocumentEmbedder(
            model=MODEL, prefix="[", suffix="]", meta_fields_to_embed=["topic"], embedding_separator=" | "
        )
        doc = Document(content="hello", meta={"topic": "ML"})
        texts = embedder._prepare_texts_to_embed([doc])
        assert texts == {doc.id: "[ML | hello]"}

    def test_prepare_input_adds_extra_body(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, extra_parameters={"truncate_prompt_tokens": 256})
        kwargs = embedder._prepare_input(["a", "b"])
        assert kwargs == {
            "model": MODEL,
            "input": ["a", "b"],
            "encoding_format": "float",
            "extra_body": {"truncate_prompt_tokens": 256},
        }

    def test_run_wrong_input_format(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL)

        with pytest.raises(TypeError, match=r"VLLMSparseDocumentEmbedder expects a list of Documents as input\."):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=r"VLLMSparseDocumentEmbedder expects a list of Documents as input\."):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}

    def test_run_batches_and_aggregates_meta(self):
        """Multi-batch run: sparse embeddings stitched back to the right docs, usage meta accumulates."""
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.embeddings.create.side_effect = [
            _fake_response([[0.1, 0.0], [0.0, 0.2]], prompt_tokens=2, total_tokens=2),
            _fake_response([[0.3, 0.4]], prompt_tokens=1, total_tokens=1),
        ]

        docs = [Document(content=f"doc-{i}") for i in range(3)]
        result = embedder.run(docs)

        assert [d.sparse_embedding for d in result["documents"]] == [
            SparseEmbedding(indices=[0], values=[0.1]),
            SparseEmbedding(indices=[1], values=[0.2]),
            SparseEmbedding(indices=[0, 1], values=[0.3, 0.4]),
        ]
        assert all(d.embedding is None for d in result["documents"])
        assert result["meta"] == {"model": "fake-model", "usage": {"prompt_tokens": 3, "total_tokens": 3}}

    def test_run_continues_on_api_error(self):
        """raise_on_failure=False: failed batches are skipped, surviving docs keep their sparse embedding."""
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.embeddings.create.side_effect = [_fake_response([[0.0, 0.1]]), _api_error()]

        result = embedder.run([Document(content="a"), Document(content="b")])

        assert result["documents"][0].sparse_embedding == SparseEmbedding(indices=[1], values=[0.1])
        assert result["documents"][1].sparse_embedding is None

    def test_run_raise_on_failure(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.embeddings.create.side_effect = _api_error()

        with pytest.raises(APIError):
            embedder.run([Document(content="a")])

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, progress_bar=True)
        embedder._async_client = MagicMock()
        embedder._async_client.embeddings.create = AsyncMock(return_value=_fake_response([[0.5, 0.0], [0.0, 0.6]]))

        docs = [Document(content="a"), Document(content="b")]
        result = await embedder.run_async(docs)

        assert [d.sparse_embedding for d in result["documents"]] == [
            SparseEmbedding(indices=[0], values=[0.5]),
            SparseEmbedding(indices=[1], values=[0.6]),
        ]

    @pytest.mark.asyncio
    async def test_run_async_continues_on_api_error(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.embeddings.create = AsyncMock(side_effect=[_api_error(), _fake_response([[0.7]])])

        result = await embedder.run_async([Document(content="a"), Document(content="b")])

        assert result["documents"][0].sparse_embedding is None
        assert result["documents"][1].sparse_embedding == SparseEmbedding(indices=[0], values=[0.7])

    @pytest.mark.asyncio
    async def test_run_async_raise_on_failure(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.embeddings.create = AsyncMock(side_effect=_api_error())

        with pytest.raises(APIError):
            await embedder.run_async([Document(content="a")])


def _sparse_dot_product(a: SparseEmbedding, b: SparseEmbedding) -> float:
    b_weights = dict(zip(b.indices, b.values, strict=True))
    return sum(value * b_weights.get(index, 0.0) for index, value in zip(a.indices, a.values, strict=True))


class TestIntegration:
    DOCS = (
        Document(content="I love cheese"),
        Document(content="Cheddar is my favorite food"),
        Document(content="A transformer is a deep learning architecture"),
    )

    @pytest.mark.integration
    def test_live_run(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, api_base_url=API_BASE_URL)

        result = embedder.run(list(self.DOCS))
        docs_with_embeddings = result["documents"]

        assert len(docs_with_embeddings) == len(self.DOCS)
        for doc in docs_with_embeddings:
            assert isinstance(doc.sparse_embedding, SparseEmbedding)
            assert len(doc.sparse_embedding.indices) > 0
            assert len(doc.sparse_embedding.indices) == len(doc.sparse_embedding.values)
            assert doc.embedding is None

        cheese, cheddar, transformer = (doc.sparse_embedding for doc in docs_with_embeddings)
        assert _sparse_dot_product(cheese, cheddar) > _sparse_dot_product(cheese, transformer)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        embedder = VLLMSparseDocumentEmbedder(model=MODEL, api_base_url=API_BASE_URL)

        result = await embedder.run_async(list(self.DOCS))
        docs_with_embeddings = result["documents"]

        assert len(docs_with_embeddings) == len(self.DOCS)
        for doc in docs_with_embeddings:
            assert isinstance(doc.sparse_embedding, SparseEmbedding)
            assert len(doc.sparse_embedding.indices) > 0
