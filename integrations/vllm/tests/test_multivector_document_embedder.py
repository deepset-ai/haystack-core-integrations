# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import math
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.embedders.vllm import VLLMMultivectorDocumentEmbedder

MODEL = os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1")
API_BASE_URL = "http://localhost:8003"
MODULE = "haystack_integrations.components.embedders.vllm.multivector_document_embedder"


def _fake_response(embeddings: list[list[list[float]]], *, status_code: int = 200):
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://localhost:8000/pooling"),
        json={
            "data": [{"index": i, "data": embedding} for i, embedding in enumerate(embeddings)],
            "model": "fake-model",
            "usage": {"prompt_tokens": len(embeddings), "total_tokens": len(embeddings)},
        },
    )


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)

        assert embedder.model == MODEL
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout is None
        assert embedder.http_client_kwargs is None
        assert embedder.extra_parameters is None
        assert embedder.concurrency_limit == 4
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_with_parameters(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model="custom-model",
            api_key=Secret.from_token("test-api-key"),
            api_base_url="https://vllm.example.test/",
            prefix="passage: ",
            suffix="!",
            batch_size=7,
            progress_bar=False,
            meta_fields_to_embed=["title", "category"],
            embedding_separator=" | ",
            timeout=3.5,
            http_client_kwargs={"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            extra_parameters={"truncate_prompt_tokens": 128, "use_activation": True},
            concurrency_limit=2,
        )

        assert embedder.model == "custom-model"
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://vllm.example.test"
        assert embedder.prefix == "passage: "
        assert embedder.suffix == "!"
        assert embedder.batch_size == 7
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["title", "category"]
        assert embedder.embedding_separator == " | "
        assert embedder.timeout == 3.5
        assert embedder.http_client_kwargs == {
            "timeout": 99,
            "headers": {"X-Custom": "value"},
            "verify": False,
        }
        assert embedder.extra_parameters == {"truncate_prompt_tokens": 128, "use_activation": True}
        assert embedder.concurrency_limit == 2

    def test_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "secret-value")
        init_parameters = {
            "model": "custom-model",
            "api_key": Secret.from_env_var("CUSTOM_VLLM_KEY"),
            "api_base_url": "https://vllm.example.test",
            "prefix": "query: ",
            "suffix": "!",
            "batch_size": 7,
            "progress_bar": False,
            "meta_fields_to_embed": ["title", "category"],
            "embedding_separator": " | ",
            "concurrency_limit": 2,
            "timeout": 12.5,
            "http_client_kwargs": {"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            "extra_parameters": {"truncate_prompt_tokens": 128, "use_activation": True},
        }
        embedder = VLLMMultivectorDocumentEmbedder(**init_parameters)

        serialized = component_to_dict(embedder, "embedder")
        assert serialized == {
            "type": f"{MODULE}.VLLMMultivectorDocumentEmbedder",
            "init_parameters": {
                **init_parameters,
                "api_key": {"type": "env_var", "env_vars": ["CUSTOM_VLLM_KEY"], "strict": True},
            },
        }
        restored = component_from_dict(VLLMMultivectorDocumentEmbedder, serialized, "embedder")
        assert component_to_dict(restored, "embedder") == serialized
        assert restored.api_key.resolve_value() == "secret-value"

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=batch_size)


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            embedder.warm_up()

    @patch(f"{MODULE}.init_http_client")
    def test_sync_lifecycle(self, init_client, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "token")
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL,
            api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
            timeout=2.5,
            http_client_kwargs={"timeout": 50, "headers": {"X-Custom": "yes"}, "verify": False},
        )
        client = MagicMock(spec=httpx.Client)
        init_client.return_value = client
        init_client.assert_not_called()

        embedder.warm_up()
        embedder.warm_up()
        init_client.assert_called_once_with(
            async_client=False,
            http_client_kwargs={
                "timeout": 2.5,
                "headers": httpx.Headers({"X-Custom": "yes", "Authorization": "Bearer token"}),
                "verify": False,
            },
        )
        assert embedder._client is client
        assert embedder._async_client is None
        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.warm_up()
        assert init_client.call_count == 2
        assert embedder._client is client

    @patch(f"{MODULE}.init_http_client")
    @pytest.mark.asyncio
    async def test_async_lifecycle(self, init_client):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL, api_key=None, http_client_kwargs={"timeout": 7, "headers": {"Accept": "x"}}
        )
        client = MagicMock(spec=httpx.AsyncClient)
        init_client.return_value = client
        init_client.assert_not_called()

        await embedder.warm_up_async()
        await embedder.warm_up_async()
        init_client.assert_called_once_with(
            async_client=True,
            http_client_kwargs={"timeout": 7, "headers": httpx.Headers({"Accept": "x"})},
        )
        assert embedder._async_client is client
        assert embedder._client is None
        await embedder.close_async()
        client.aclose.assert_awaited_once_with()
        assert embedder._async_client is None
        await embedder.warm_up_async()
        assert init_client.call_count == 2
        assert embedder._async_client is client

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

        sync_client = MagicMock(spec=httpx.Client)
        async_client = MagicMock(spec=httpx.AsyncClient)
        embedder._client = sync_client
        embedder._async_client = async_client

        embedder.close()
        assert embedder._client is None
        assert embedder._async_client is async_client
        async_client.aclose.assert_not_awaited()
        await embedder.close_async()
        assert embedder._async_client is None
        embedder.close()
        await embedder.close_async()
        sync_client.close.assert_called_once_with()
        async_client.aclose.assert_awaited_once_with()


class TestRun:
    @pytest.mark.parametrize("documents", ["document", [Document(content="valid"), 1], [None]])
    def test_run_wrong_input_format(self, documents):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)
        with patch.object(embedder, "warm_up") as warm_up:
            with pytest.raises(TypeError, match="expects a list containing only Documents"):
                embedder.run(documents)
        warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_empty_documents(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)
        with (
            patch.object(embedder, "warm_up") as warm_up,
            patch.object(embedder, "warm_up_async", new_callable=AsyncMock) as warm_up_async,
        ):
            assert embedder.run([]) == {"documents": []}
            assert await embedder.run_async([]) == {"documents": []}
        warm_up.assert_not_called()
        warm_up_async.assert_not_awaited()

    def test_run_with_mock(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL,
            prefix="[",
            suffix="]",
            meta_fields_to_embed=["title", "missing", "priority"],
            embedding_separator=" | ",
            progress_bar=False,
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "use_activation": True},
        )
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = _fake_response([[[1.0, 1.5], [2.0, 2.5]]])
        metadata = {"title": "Title", "priority": 2, "missing": None}
        document = Document(content="body", meta=metadata, embedding=[0.9, 0.8], multivector_embedding=[[0.1, 0.2]])

        result = embedder.run([document])

        embedder._client.post.assert_called_once_with(
            "http://localhost:8000/pooling",
            json={
                "use_activation": True,
                "model": MODEL,
                "input": ["[Title | 2 | body]"],
                "task": "token_embed",
            },
        )
        copied = result["documents"][0]
        assert copied is not document
        assert copied.id == document.id
        assert copied.content == "body"
        assert copied.meta == {"title": "Title", "priority": 2, "missing": None}
        assert copied.embedding == [0.9, 0.8]
        assert copied.multivector_embedding == [[1.0, 1.5], [2.0, 2.5]]
        assert document.multivector_embedding == [[0.1, 0.2]]

    def test_run_batches(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.side_effect = [
            _fake_response([[[0.1]], [[0.2]]]),
            _fake_response([[[0.3]]]),
        ]
        documents = [Document(content=f"doc-{index}") for index in range(3)]

        result = embedder.run(documents)

        assert [doc.multivector_embedding for doc in result["documents"]] == [[[0.1]], [[0.2]], [[0.3]]]
        assert [call.kwargs["json"]["input"] for call in embedder._client.post.call_args_list] == [
            ["doc-0", "doc-1"],
            ["doc-2"],
        ]

    def test_run_with_duplicate_document_ids(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.side_effect = [
            _fake_response([[[0.1]]]),
            _fake_response([[[0.2]]]),
        ]
        documents = [Document(id="duplicate", content="first"), Document(id="duplicate", content="second")]

        result = embedder.run(documents)

        assert [doc.multivector_embedding for doc in result["documents"]] == [[[0.1]], [[0.2]]]

    def test_run_stops_on_http_error(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        response = _fake_response([], status_code=503)
        embedder._client.post.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            embedder.run([Document(content="failed"), Document(content="not processed")])

        embedder._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)
        responses = [
            _fake_response([[[0.1]], [[0.2]]]),
            _fake_response([[[0.3]]]),
        ]
        embedder._async_client.post = AsyncMock(side_effect=responses)
        originals = [Document(content=f"doc-{index}", embedding=[float(index)]) for index in range(3)]

        result = await embedder.run_async(originals)

        assert [doc.multivector_embedding for doc in result["documents"]] == [[[0.1]], [[0.2]], [[0.3]]]
        assert [doc.embedding for doc in result["documents"]] == [[0.0], [1.0], [2.0]]
        assert all(copied is not original for copied, original in zip(result["documents"], originals, strict=True))
        assert all(original.multivector_embedding is None for original in originals)
        assert embedder._async_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_run_async_raises_on_http_error(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, progress_bar=False)
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)
        response = _fake_response([], status_code=500)
        embedder._async_client.post = AsyncMock(return_value=response)
        with pytest.raises(httpx.HTTPStatusError):
            await embedder.run_async([Document(content="failed")])


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, api_base_url=API_BASE_URL, batch_size=2, timeout=60)
        docs = [
            Document(content="I love cheese"),
            Document(content="Cheddar is my favorite food"),
            Document(content="A transformer is a deep learning architecture"),
        ]

        try:
            result = embedder.run(docs)
        finally:
            embedder.close()
        docs_with_embeddings = result["documents"]

        assert [doc.id for doc in docs_with_embeddings] == [doc.id for doc in docs]
        dimensions = set()
        for doc in docs_with_embeddings:
            embedding = doc.multivector_embedding
            assert embedding and embedding[0]
            dimensions.update(len(vector) for vector in embedding)
            assert all(isinstance(value, float) and math.isfinite(value) for vector in embedding for value in vector)
        assert len(dimensions) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL, api_base_url=API_BASE_URL, batch_size=1, concurrency_limit=2, timeout=60
        )
        docs = [
            Document(content="I love cheese"),
            Document(content="Cheddar is my favorite food"),
            Document(content="A transformer is a deep learning architecture"),
        ]

        try:
            result = await embedder.run_async(docs)
        finally:
            await embedder.close_async()
        docs_with_embeddings = result["documents"]

        assert [doc.id for doc in docs_with_embeddings] == [doc.id for doc in docs]
        dimensions = set()
        for doc in docs_with_embeddings:
            embedding = doc.multivector_embedding
            assert embedding and embedding[0]
            dimensions.update(len(vector) for vector in embedding)
            assert all(isinstance(value, float) and math.isfinite(value) for vector in embedding for value in vector)
        assert len(dimensions) == 1
