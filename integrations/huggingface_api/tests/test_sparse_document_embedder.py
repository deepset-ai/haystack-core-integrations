# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret

from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseDocumentEmbedder

API_BASE_URL = "http://localhost:8080"


def sparse_response(data):
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    return response


class TestHuggingFaceAPISparseDocumentEmbedder:
    @pytest.mark.parametrize("api_base_url", ["not-a-url", "file:///path", "localhost:8080"])
    def test_init_rejects_invalid_api_base_url(self, api_base_url):
        with pytest.raises(ValueError, match="api_base_url must be a valid HTTP URL"):
            HuggingFaceAPISparseDocumentEmbedder(api_base_url=api_base_url)

    @pytest.mark.parametrize(("parameter", "value"), [("batch_size", 0), ("batch_size", -2), ("concurrency_limit", 0)])
    def test_init_rejects_non_positive_numeric_options(self, parameter, value):
        with pytest.raises(ValueError, match=f"{parameter} must be > 0"):
            HuggingFaceAPISparseDocumentEmbedder(**{parameter: value})

    def test_init_defaults(self):
        embedder = HuggingFaceAPISparseDocumentEmbedder()

        assert embedder.api_base_url == "http://localhost:8080"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout == 30.0
        assert embedder.headers == {}
        assert embedder.concurrency_limit == 4

    def test_to_dict_and_from_dict_preserve_configuration_and_secret(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_HF_TOKEN", "secret")
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="https://tei.example.test",
            token=Secret.from_env_var("CUSTOM_HF_TOKEN"),
            prefix="passage: ",
            suffix="!",
            batch_size=2,
            progress_bar=False,
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
            timeout=None,
            headers={"X-Test": "yes"},
            concurrency_limit=3,
        )

        data = embedder.to_dict()
        restored = HuggingFaceAPISparseDocumentEmbedder.from_dict(data)

        assert data["init_parameters"]["token"] == {
            "type": "env_var",
            "env_vars": ["CUSTOM_HF_TOKEN"],
            "strict": True,
        }
        assert data["init_parameters"] == {
            "api_base_url": "https://tei.example.test",
            "token": {"type": "env_var", "env_vars": ["CUSTOM_HF_TOKEN"], "strict": True},
            "prefix": "passage: ",
            "suffix": "!",
            "batch_size": 2,
            "progress_bar": False,
            "meta_fields_to_embed": ["title"],
            "embedding_separator": " | ",
            "timeout": None,
            "headers": {"X-Test": "yes"},
            "concurrency_limit": 3,
        }
        assert restored.api_base_url == embedder.api_base_url
        assert restored.prefix == "passage: "
        assert restored.suffix == "!"
        assert restored.batch_size == 2
        assert restored.progress_bar is False
        assert restored.meta_fields_to_embed == ["title"]
        assert restored.embedding_separator == " | "
        assert restored.timeout is None
        assert restored.headers == {"X-Test": "yes"}
        assert restored.concurrency_limit == 3
        assert restored.token is not None and restored.token.resolve_value() == "secret"
        assert "_client" not in data["init_parameters"]
        assert "_async_client" not in data["init_parameters"]

    def test_token_secret_cannot_be_serialized(self):
        embedder = HuggingFaceAPISparseDocumentEmbedder(token=Secret.from_token("do-not-serialize"))

        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            embedder.to_dict()

    def test_prepare_texts_handles_metadata_prefix_suffix_and_missing_content(self):
        documents = [
            Document(content="body", meta={"title": "Title", "priority": 3, "ignored": "no"}),
            Document(content=None, meta={"title": None, "priority": 0}),
            Document(content="only body", meta={}),
        ]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            prefix="<p>", suffix="</p>", meta_fields_to_embed=["title", "priority"], embedding_separator=" | "
        )

        assert embedder._prepare_texts_to_embed(documents) == [
            "<p>Title | 3 | body</p>",
            "<p>0 | </p>",
            "<p>only body</p>",
        ]

    @pytest.mark.parametrize("documents", [None, "document", [1, 2], [Document(content="valid"), "invalid"]])
    def test_run_rejects_invalid_and_mixed_inputs(self, documents):
        with pytest.raises(TypeError, match="expects a list of Documents"):
            HuggingFaceAPISparseDocumentEmbedder(progress_bar=False).run(documents)

    @pytest.mark.asyncio
    async def test_run_async_rejects_mixed_inputs(self):
        with pytest.raises(TypeError, match="expects a list of Documents"):
            await HuggingFaceAPISparseDocumentEmbedder(progress_bar=False).run_async([Document(content="ok"), None])

    def test_empty_list_returns_without_http_request(self):
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)

        result = embedder.run([])

        assert result == {"documents": []}
        embedder._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_async_returns_without_http_request(self):
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        embedder._async_client = MagicMock(spec=httpx.AsyncClient)

        result = await embedder.run_async([])

        assert result == {"documents": []}
        embedder._async_client.post.assert_not_called()

    def test_run_batches_requests_preserves_order_and_copies_documents(self):
        documents = [Document(content=f"doc {number}", meta={"number": number}) for number in range(5)]
        responses = [
            sparse_response([[{"index": 10, "value": 1}], [{"index": 11, "value": 2}]]),
            sparse_response([[{"index": 12, "value": 3}], [{"index": 13, "value": 4}]]),
            sparse_response([[{"index": 14, "value": 5}]]),
        ]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="http://tei:80/root/",
            token=Secret.from_token("token"),
            prefix="passage: ",
            batch_size=2,
            progress_bar=False,
            timeout=6,
            headers={"X-Tenant": "tenant"},
        )

        client = MagicMock(spec=httpx.Client)
        client.post.side_effect = responses
        embedder._client = client

        result = embedder.run(documents)

        assert client.post.call_args_list == [
            call("embed_sparse", json={"inputs": ["passage: doc 0", "passage: doc 1"]}),
            call("embed_sparse", json={"inputs": ["passage: doc 2", "passage: doc 3"]}),
            call("embed_sparse", json={"inputs": ["passage: doc 4"]}),
        ]
        output = result["documents"]
        assert [document.sparse_embedding.indices for document in output] == [[10], [11], [12], [13], [14]]
        assert [document.sparse_embedding.values for document in output] == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert all(new is not original for original, new in zip(documents, output, strict=True))
        assert all(original.sparse_embedding is None for original in documents)
        assert [document.meta for document in output] == [document.meta for document in documents]

    @pytest.mark.asyncio
    async def test_run_async_batches_with_one_client_and_preserves_batch_order(self):
        async def post(_url, *, json):
            inputs = json["inputs"]
            # Let the second batch finish first; gather must still preserve input order.
            await asyncio.sleep(0.01 if inputs[0] == "doc 0" else 0)
            offset = int(inputs[0].split()[-1])
            return sparse_response([[{"index": offset + position, "value": 1}] for position, _ in enumerate(inputs)])

        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(side_effect=post)
        documents = [Document(content=f"doc {number}") for number in range(4)]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="https://tei.test/", batch_size=2, progress_bar=False, timeout=None, headers={"X-Test": "yes"}
        )
        embedder._async_client = client

        result = await embedder.run_async(documents)

        assert client.post.await_args_list == [
            call("embed_sparse", json={"inputs": ["doc 0", "doc 1"]}),
            call("embed_sparse", json={"inputs": ["doc 2", "doc 3"]}),
        ]
        assert [document.sparse_embedding.indices for document in result["documents"]] == [[0], [1], [2], [3]]

    @pytest.mark.asyncio
    async def test_async_concurrency_limit_is_respected(self):
        active = 0
        maximum_active = 0

        async def embed_batch(**kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [SparseEmbedding(indices=[int(text)], values=[1.0]) for text in kwargs["inputs"]]

        embedder = HuggingFaceAPISparseDocumentEmbedder(batch_size=1, concurrency_limit=2, progress_bar=False)
        with patch(
            "haystack_integrations.components.embedders.huggingface_api.sparse_document_embedder._embed_sparse_async",
            side_effect=embed_batch,
        ):
            embeddings = await embedder._embed_batches_async(["0", "1", "2", "3"])

        assert maximum_active == 2
        assert [embedding.indices for embedding in embeddings] == [[0], [1], [2], [3]]

    def test_run_rejects_response_with_wrong_embedding_count(self):
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])

        with pytest.raises(ValueError, match="Expected one sparse embedding per input"):
            embedder.run([Document(content="one"), Document(content="two")])

    def test_run_propagates_http_error(self):
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        response = httpx.Response(500, request=request)
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = response

        with pytest.raises(httpx.HTTPStatusError):
            embedder.run([Document(content="text")])

    @pytest.mark.integration
    def test_live_run_tei(self):
        documents = [Document(content="sparse retrieval"), Document(content="dense retrieval")]
        result = HuggingFaceAPISparseDocumentEmbedder(api_base_url=API_BASE_URL, progress_bar=False).run(documents)

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(documents)
        for document in documents_with_embeddings:
            assert isinstance(document.sparse_embedding, SparseEmbedding)
            assert document.sparse_embedding.indices
