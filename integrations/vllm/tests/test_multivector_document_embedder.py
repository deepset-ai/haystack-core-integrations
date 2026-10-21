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
from haystack_integrations.components.embedders.vllm.multivector_document_embedder import (
    VLLMMultivectorDocumentEmbedder as DirectVLLMMultivectorDocumentEmbedder,
)

MODEL = "jinaai/jina-colbert-v2"
MODULE = "haystack_integrations.components.embedders.vllm.multivector_document_embedder"


def _response(data, *, status_code=200):
    request = httpx.Request("POST", "http://localhost:8000/pooling")
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = {
        "data": data,
        "model": "served-model",
        "usage": {"prompt_tokens": len(data), "total_tokens": len(data)},
    }
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "request failed", request=request, response=httpx.Response(status_code, request=request)
        )
    return response


def _item(index, value):
    return {"index": index, "data": [[value, value + 0.5], [value + 1, value + 1.5]]}


class TestInitializationAndSerialization:
    def test_default_constructor_state_and_export(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)

        assert DirectVLLMMultivectorDocumentEmbedder is VLLMMultivectorDocumentEmbedder
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
        assert embedder.raise_on_failure is False
        assert embedder.extra_parameters is None
        assert embedder.concurrency_limit == 4
        assert embedder._client is None
        assert embedder._async_client is None
        assert set(embedder.run._output_types_cache) == {"documents"}
        assert embedder.run._output_types_cache["documents"].type == list[Document]
        assert set(embedder.run_async._output_types_cache) == {"documents"}
        assert embedder.run_async._output_types_cache["documents"].type == list[Document]

    def test_non_default_constructor_and_serialization_round_trip(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "secret-value")
        embedder = VLLMMultivectorDocumentEmbedder(
            model="custom-model",
            api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
            api_base_url="https://vllm.example.test/",
            prefix="passage: ",
            suffix="!",
            batch_size=7,
            progress_bar=False,
            meta_fields_to_embed=["title", "category"],
            embedding_separator=" | ",
            timeout=3.5,
            http_client_kwargs={"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
            raise_on_failure=True,
            extra_parameters={"truncate_prompt_tokens": 128, "normalize": True},
            concurrency_limit=2,
        )

        serialized = component_to_dict(embedder, "embedder")
        assert serialized == {
            "type": f"{MODULE}.VLLMMultivectorDocumentEmbedder",
            "init_parameters": {
                "model": "custom-model",
                "api_key": {"type": "env_var", "env_vars": ["CUSTOM_VLLM_KEY"], "strict": True},
                "api_base_url": "https://vllm.example.test",
                "prefix": "passage: ",
                "suffix": "!",
                "batch_size": 7,
                "progress_bar": False,
                "meta_fields_to_embed": ["title", "category"],
                "embedding_separator": " | ",
                "timeout": 3.5,
                "http_client_kwargs": {"timeout": 99, "headers": {"X-Custom": "value"}, "verify": False},
                "raise_on_failure": True,
                "extra_parameters": {"truncate_prompt_tokens": 128, "normalize": True},
                "concurrency_limit": 2,
            },
        }

        restored = component_from_dict(VLLMMultivectorDocumentEmbedder, serialized, "embedder")
        assert restored.model == "custom-model"
        assert restored.api_key == Secret.from_env_var("CUSTOM_VLLM_KEY")
        assert restored.api_key.resolve_value() == "secret-value"
        assert restored.api_base_url == "https://vllm.example.test"
        assert restored.prefix == "passage: "
        assert restored.suffix == "!"
        assert restored.batch_size == 7
        assert restored.progress_bar is False
        assert restored.meta_fields_to_embed == ["title", "category"]
        assert restored.embedding_separator == " | "
        assert restored.timeout == 3.5
        assert restored.http_client_kwargs == {
            "timeout": 99,
            "headers": {"X-Custom": "value"},
            "verify": False,
        }
        assert restored.raise_on_failure is True
        assert restored.extra_parameters == {"truncate_prompt_tokens": 128, "normalize": True}
        assert restored.concurrency_limit == 2

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=batch_size)


class TestComponentLifecycle:
    def test_sync_warm_up_is_lazy_idempotent_and_close(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_VLLM_KEY", "token")
        client = MagicMock(spec=httpx.Client)
        with patch(f"{MODULE}.init_http_client", return_value=client) as init_client:
            embedder = VLLMMultivectorDocumentEmbedder(
                model=MODEL,
                api_key=Secret.from_env_var("CUSTOM_VLLM_KEY"),
                timeout=2.5,
                http_client_kwargs={"timeout": 50, "headers": {"X-Custom": "yes"}, "verify": False},
            )
            init_client.assert_not_called()
            embedder.warm_up()
            embedder.warm_up()

        init_client.assert_called_once()
        kwargs = init_client.call_args.kwargs
        assert kwargs["async_client"] is False
        assert kwargs["http_client_kwargs"]["timeout"] == 2.5
        assert kwargs["http_client_kwargs"]["verify"] is False
        headers = kwargs["http_client_kwargs"]["headers"]
        assert headers["Authorization"] == "Bearer token"
        assert headers["X-Custom"] == "yes"

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.close()

    @pytest.mark.asyncio
    async def test_async_warm_up_is_lazy_idempotent_and_close(self):
        client = MagicMock(spec=httpx.AsyncClient)
        client.aclose = AsyncMock()
        with patch(f"{MODULE}.init_http_client", return_value=client) as init_client:
            embedder = VLLMMultivectorDocumentEmbedder(
                model=MODEL, api_key=None, http_client_kwargs={"timeout": 7, "headers": {"Accept": "x"}}
            )
            await embedder.warm_up_async()
            await embedder.warm_up_async()

        init_client.assert_called_once()
        kwargs = init_client.call_args.kwargs
        assert kwargs["async_client"] is True
        assert kwargs["http_client_kwargs"]["timeout"] == 7
        assert "Authorization" not in kwargs["http_client_kwargs"]["headers"]
        assert kwargs["http_client_kwargs"]["headers"]["Accept"] == "x"
        assert embedder._client is None

        await embedder.close_async()
        client.aclose.assert_awaited_once_with()
        assert embedder._async_client is None
        await embedder.close_async()

    def test_secret_is_resolved_only_during_warm_up(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_KEY", raising=False)
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_KEY"):
            embedder.warm_up()


class TestRun:
    def test_prepare_texts_to_embed(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL,
            prefix="[",
            suffix="]",
            meta_fields_to_embed=["title", "missing", "priority"],
            embedding_separator=" | ",
        )
        document = Document(content="body", meta={"title": "Title", "priority": 2, "missing": None})

        assert embedder._prepare_texts_to_embed([document]) == ["[Title | 2 | body]"]

    def test_prepare_input(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL,
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "normalize": True},
        )

        assert embedder._prepare_input(["a", "b"]) == {
            "normalize": True,
            "model": MODEL,
            "input": ["a", "b"],
            "task": "token_embed",
        }

    @pytest.mark.parametrize("documents", ["document", [Document(content="valid"), 1], [None]])
    def test_validates_all_documents_before_warm_up(self, documents):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)
        with patch.object(embedder, "warm_up") as warm_up:
            with pytest.raises(TypeError, match="expects a list containing only Documents"):
                embedder.run(documents)
        warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_does_not_warm_up_sync_or_async(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL)
        with (
            patch.object(embedder, "warm_up") as warm_up,
            patch.object(embedder, "warm_up_async", new_callable=AsyncMock) as warm_up_async,
        ):
            assert embedder.run([]) == {"documents": []}
            assert await embedder.run_async([]) == {"documents": []}
        warm_up.assert_not_called()
        warm_up_async.assert_not_awaited()

    def test_metadata_text_payload_and_reserved_parameter_precedence(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL,
            prefix="[",
            suffix="]",
            meta_fields_to_embed=["title", "missing", "priority"],
            embedding_separator=" | ",
            progress_bar=False,
            extra_parameters={"model": "wrong", "input": "wrong", "task": "wrong", "normalize": True},
        )
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = _response([_item(0, 1.0)])
        document = Document(content="body", meta={"title": "Title", "priority": 2, "missing": None})

        embedder.run([document])

        embedder._client.post.assert_called_once_with(
            "http://localhost:8000/pooling",
            json={
                "normalize": True,
                "model": MODEL,
                "input": ["[Title | 2 | body]"],
                "task": "token_embed",
            },
        )
        embedder._client.post.return_value.raise_for_status.assert_called_once_with()

    def test_batches_map_out_of_order_provider_indices(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.side_effect = [
            _response([_item(1, 20.0), _item(0, 10.0)]),
            _response([_item(0, 30.0)]),
        ]
        documents = [Document(content=f"doc-{index}") for index in range(3)]

        result = embedder.run(documents)

        vectors = [document.meta["_multivector_embedding"] for document in result["documents"]]
        assert vectors == [
            [[10.0, 10.5], [11.0, 11.5]],
            [[20.0, 20.5], [21.0, 21.5]],
            [[30.0, 30.5], [31.0, 31.5]],
        ]
        assert set(result) == {"documents"}
        assert [call.kwargs["json"]["input"] for call in embedder._client.post.call_args_list] == [
            ["doc-0", "doc-1"],
            ["doc-2"],
        ]

    def test_duplicate_document_ids_are_mapped_by_position(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.side_effect = [_response([_item(0, 1.0)]), _response([_item(0, 2.0)])]
        documents = [Document(id="duplicate", content="first"), Document(id="duplicate", content="second")]

        result = embedder.run(documents)

        assert [document.meta["_multivector_embedding"][0][0] for document in result["documents"]] == [1.0, 2.0]

    def test_returns_copies_without_mutating_originals_and_preserves_dense_embedding(self):
        original_meta = {"source": "test", "nested": {"value": 1}}
        original = Document(content="body", meta=original_meta, embedding=[0.9, 0.8])
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = _response([_item(0, 1.0)])

        copied = embedder.run([original])["documents"][0]

        assert copied is not original
        assert copied.id == original.id
        assert copied.embedding == [0.9, 0.8]
        assert copied.meta["_multivector_embedding"] == [[1.0, 1.5], [2.0, 2.5]]
        assert original.meta == {"source": "test", "nested": {"value": 1}}
        assert original_meta == {"source": "test", "nested": {"value": 1}}
        assert "_multivector_embedding" not in original.meta

    def test_failed_batch_continues_and_successful_batch_is_returned(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=MODEL, batch_size=1, progress_bar=False, raise_on_failure=False
        )
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.side_effect = [_response([], status_code=503), _response([_item(0, 2.0)])]

        result = embedder.run([Document(content="failed"), Document(content="successful")])

        assert "_multivector_embedding" not in result["documents"][0].meta
        assert result["documents"][1].meta["_multivector_embedding"] == [[2.0, 2.5], [3.0, 3.5]]
        assert set(result) == {"documents"}

    def test_failed_batch_raises_when_configured(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, progress_bar=False, raise_on_failure=True)
        embedder._client = MagicMock(spec=httpx.Client)
        response = _response([], status_code=503)
        embedder._client.post.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            embedder.run([Document(content="failed")])

        embedder._client.post.assert_called_once_with(
            "http://localhost:8000/pooling",
            json={"model": MODEL, "input": ["failed"], "task": "token_embed"},
        )
        response.raise_for_status.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_async_has_batch_mapping_and_copying_parity(self):
        embedder = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)
        responses = [
            _response([_item(1, 2.0), _item(0, 1.0)]),
            _response([_item(0, 3.0)]),
        ]
        embedder._async_client.post = AsyncMock(side_effect=responses)
        originals = [Document(content=f"doc-{index}", embedding=[float(index)]) for index in range(3)]

        result = await embedder.run_async(originals)

        assert [doc.meta["_multivector_embedding"][0][0] for doc in result["documents"]] == [1.0, 2.0, 3.0]
        assert [doc.embedding for doc in result["documents"]] == [[0.0], [1.0], [2.0]]
        assert all(copied is not original for copied, original in zip(result["documents"], originals, strict=True))
        assert all("_multivector_embedding" not in original.meta for original in originals)
        assert set(result) == {"documents"}
        assert embedder._async_client.post.await_count == 2
        for response in responses:
            response.raise_for_status.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_async_failed_batch_continue_and_raise(self):
        continuing = VLLMMultivectorDocumentEmbedder(model=MODEL, batch_size=1, progress_bar=False)
        continuing._async_client = MagicMock(spec=httpx.AsyncClient)
        failed_response = _response([], status_code=500)
        successful_response = _response([_item(0, 2.0)])
        continuing._async_client.post = AsyncMock(side_effect=[failed_response, successful_response])

        result = await continuing.run_async([Document(content="failed"), Document(content="successful")])

        assert "_multivector_embedding" not in result["documents"][0].meta
        assert result["documents"][1].meta["_multivector_embedding"] == [[2.0, 2.5], [3.0, 3.5]]
        failed_response.raise_for_status.assert_called_once_with()
        successful_response.raise_for_status.assert_called_once_with()

        raising = VLLMMultivectorDocumentEmbedder(model=MODEL, progress_bar=False, raise_on_failure=True)
        raising._async_client = MagicMock(spec=httpx.AsyncClient)
        raising_response = _response([], status_code=500)
        raising._async_client.post = AsyncMock(return_value=raising_response)
        with pytest.raises(httpx.HTTPStatusError):
            await raising.run_async([Document(content="failed")])
        raising_response.raise_for_status.assert_called_once_with()


@pytest.mark.integration
class TestIntegration:
    @staticmethod
    def _documents() -> list[Document]:
        return [
            Document(id="rome", content="Rome is the capital of Italy.", meta={"source": "cities"}, embedding=[0.1]),
            Document(id="paris", content="Paris is the capital of France.", meta={"source": "cities"}, embedding=[0.2]),
            Document(
                id="berlin", content="Berlin is the capital of Germany.", meta={"source": "cities"}, embedding=[0.3]
            ),
        ]

    @staticmethod
    def _assert_documents(result: dict[str, list[Document]], originals: list[Document]) -> None:
        assert set(result) == {"documents"}
        embedded = result["documents"]
        assert [document.id for document in embedded] == [document.id for document in originals]
        assert len(embedded) == len(originals)

        dimensions = set()
        for copied, original in zip(embedded, originals, strict=True):
            assert copied is not original
            assert copied.meta is not original.meta
            assert copied.id == original.id
            assert copied.content == original.content
            assert copied.embedding == original.embedding
            assert {
                key: value for key, value in copied.meta.items() if key != "_multivector_embedding"
            } == original.meta

            multivector = copied.meta["_multivector_embedding"]
            assert multivector
            dimensions.update(len(vector) for vector in multivector)
            assert all(isinstance(value, float) and math.isfinite(value) for vector in multivector for value in vector)

        assert len(dimensions) == 1
        assert dimensions.pop() > 0
        assert [document.id for document in originals] == ["rome", "paris", "berlin"]
        assert [document.content for document in originals] == [
            "Rome is the capital of Italy.",
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ]
        assert [document.meta for document in originals] == [{"source": "cities"}] * 3
        assert [document.embedding for document in originals] == [[0.1], [0.2], [0.3]]

    def test_run(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1"),
            api_base_url="http://localhost:8003",
            batch_size=2,
            progress_bar=False,
            timeout=60,
        )
        originals = self._documents()

        try:
            result = embedder.run(originals)
            self._assert_documents(result, originals)
        finally:
            embedder.close()

    @pytest.mark.asyncio
    async def test_run_async(self):
        embedder = VLLMMultivectorDocumentEmbedder(
            model=os.environ.get("VLLM_MULTIVECTOR_MODEL", "answerdotai/answerai-colbert-small-v1"),
            api_base_url="http://localhost:8003",
            batch_size=1,
            progress_bar=False,
            timeout=60,
            concurrency_limit=2,
        )
        originals = self._documents()

        try:
            result = await embedder.run_async(originals)
            self._assert_documents(result, originals)
        finally:
            await embedder.close_async()
