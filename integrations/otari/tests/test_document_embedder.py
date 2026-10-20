# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack import Document
from haystack.utils import Secret
from openai import APIError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.otari import OtariDocumentEmbedder

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "openai:text-embedding-3-small"
DEFAULT_API_BASE_URL = "https://api.otari.ai/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.embedders.otari.document_embedder.OtariDocumentEmbedder"


def _embedding_response(value: float, cost_usd: str | None = None) -> CreateEmbeddingResponse:
    usage = Usage(prompt_tokens=10, total_tokens=10)
    if cost_usd is not None:
        usage = Usage(prompt_tokens=10, total_tokens=10, cost_usd=cost_usd, pricing_source="organization")
    return CreateEmbeddingResponse(
        data=[Embedding(embedding=[value, value], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=usage,
    )


THREE_DOCS = [Document(content="a"), Document(content="b"), Document(content="c")]


class TestOtariDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var("OTARI_API_KEY")
        assert embedder.model == DEFAULT_MODEL
        assert embedder.dimensions is None
        assert embedder.api_base_url == DEFAULT_API_BASE_URL
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.raise_on_failure is False

    def test_warm_up_warns_once_about_eu_key_on_default_url(self, caplog):
        component = OtariDocumentEmbedder(api_key=Secret.from_token("otk_v1_eu_test"))
        component.warm_up()
        component.warm_up()
        assert caplog.text.count("belongs to otari.ai's EU region") == 1

    @pytest.mark.asyncio
    async def test_warm_up_async_warns_once_about_eu_key_on_default_url(self, caplog):
        component = OtariDocumentEmbedder(api_key=Secret.from_token("otk_v1_eu_test"))
        await component.warm_up_async()
        await component.warm_up_async()
        assert caplog.text.count("belongs to otari.ai's EU region") == 1

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        assert OtariDocumentEmbedder().to_dict() == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "dimensions": None,
                "api_base_url": DEFAULT_API_BASE_URL,
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
            },
        }

    def test_to_dict_from_dict_round_trip_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        embedder = OtariDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral:mistral-embed",
            dimensions=256,
            api_base_url="https://otari.example.com/api/v1",
            prefix="passage: ",
            suffix=" end",
            batch_size=8,
            progress_bar=False,
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
            raise_on_failure=True,
        )
        data = embedder.to_dict()
        assert "organization" not in data["init_parameters"]

        restored = OtariDocumentEmbedder.from_dict(data)
        assert restored.api_key == Secret.from_env_var("ENV_VAR")
        assert restored.model == "mistral:mistral-embed"
        assert restored.dimensions == 256
        assert restored.api_base_url == "https://otari.example.com/api/v1"
        assert restored.prefix == "passage: "
        assert restored.suffix == " end"
        assert restored.batch_size == 8
        assert restored.progress_bar is False
        assert restored.meta_fields_to_embed == ["title"]
        assert restored.embedding_separator == " | "
        assert restored.timeout == 10
        assert restored.max_retries == 2
        assert restored.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert restored.raise_on_failure is True

    def test_run(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(meta_fields_to_embed=["title"], progress_bar=False)
        docs = [
            Document(content="I love pizza!", meta={"title": "Food"}),
            Document(content="Haystack builds pipelines."),
        ]
        with patch("openai.resources.embeddings.Embeddings.create") as mock_create:
            mock_create.return_value = CreateEmbeddingResponse(
                data=[
                    Embedding(embedding=[0.1, 0.2], index=0, object="embedding"),
                    Embedding(embedding=[0.3, 0.4], index=1, object="embedding"),
                ],
                model="text-embedding-3-small",
                object="list",
                usage=Usage(prompt_tokens=10, total_tokens=10),
            )
            result = embedder.run(docs)

        _, kwargs = mock_create.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["input"] == ["Food\nI love pizza!", "Haystack builds pipelines."]
        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.2], [0.3, 0.4]]
        assert result["meta"]["usage"] == {"prompt_tokens": 10, "total_tokens": 10}

    def test_run_sums_cost_over_batches(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(batch_size=1, progress_bar=False)
        responses = [
            _embedding_response(0.1, "0.000016"),
            _embedding_response(0.2, "0.000017"),
            _embedding_response(0.3, "0.000016"),
        ]
        with patch("openai.resources.embeddings.Embeddings.create", side_effect=responses) as mock_create:
            result = embedder.run(THREE_DOCS)

        assert mock_create.call_count == 3
        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        assert result["meta"]["usage"] == {
            "prompt_tokens": 30,
            "total_tokens": 30,
            "cost_usd": "0.000049",
            "pricing_source": "organization",
        }

    @pytest.mark.asyncio
    async def test_run_async_sums_cost_over_batches(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(batch_size=1, progress_bar=False)
        responses = [
            _embedding_response(0.1, "0.000016"),
            _embedding_response(0.2, "0.000017"),
            _embedding_response(0.3, "0.000016"),
        ]
        with patch("openai.resources.embeddings.AsyncEmbeddings.create", new_callable=AsyncMock, side_effect=responses):
            result = await embedder.run_async(THREE_DOCS)

        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        assert result["meta"]["usage"]["cost_usd"] == "0.000049"
        assert result["meta"]["usage"]["total_tokens"] == 30

    def test_run_drops_cost_when_a_batch_has_none(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(batch_size=1, progress_bar=False)
        responses = [
            _embedding_response(0.1, "0.000016"),
            _embedding_response(0.2),
            _embedding_response(0.3, "0.000016"),
        ]
        with patch("openai.resources.embeddings.Embeddings.create", side_effect=responses):
            result = embedder.run(THREE_DOCS)

        # a partial sum would understate the cost, so neither cost field is reported
        assert result["meta"]["usage"] == {"prompt_tokens": 30, "total_tokens": 30}

    def test_run_skips_a_failed_batch(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(batch_size=1, progress_bar=False)
        error = APIError("boom", request=httpx.Request("POST", "https://api.otari.ai/api/v1/embeddings"), body=None)
        responses = [_embedding_response(0.1, "0.000016"), error, _embedding_response(0.3, "0.000016")]
        with patch("openai.resources.embeddings.Embeddings.create", side_effect=responses):
            result = embedder.run(THREE_DOCS)

        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.1], None, [0.3, 0.3]]
        assert result["meta"]["usage"]["cost_usd"] == "0.000032"

    def test_run_raises_on_a_failed_batch_with_raise_on_failure(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(batch_size=1, progress_bar=False, raise_on_failure=True)
        error = APIError("boom", request=httpx.Request("POST", "https://api.otari.ai/api/v1/embeddings"), body=None)
        with patch("openai.resources.embeddings.Embeddings.create", side_effect=[error]), pytest.raises(APIError):
            embedder.run(THREE_DOCS)

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_sums_cost_over_batches(self):
        # about 700 tokens per document, so that each batch costs at least a micro-dollar
        docs = [Document(content=f"Document {i}. " + "Haystack builds LLM applications. " * 120) for i in range(3)]
        single = OtariDocumentEmbedder(api_base_url=OTARI_API_BASE_URL, batch_size=3, progress_bar=False).run(docs)
        if "cost_usd" not in single["meta"]["usage"]:
            pytest.skip("The gateway doesn't price this embedding model")
        multi_batch = OtariDocumentEmbedder(api_base_url=OTARI_API_BASE_URL, batch_size=1, progress_bar=False).run(docs)

        single_cost = Decimal(single["meta"]["usage"]["cost_usd"])
        multi_batch_cost = Decimal(multi_batch["meta"]["usage"]["cost_usd"])
        # each batch's cost is rounded to a micro-dollar
        assert abs(multi_batch_cost - single_cost) <= Decimal("0.000003")
        assert multi_batch["meta"]["usage"]["total_tokens"] == single["meta"]["usage"]["total_tokens"]

    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self):
        docs = [
            Document(content="I love pizza!"),
            Document(content="Haystack is an open-source framework for building RAG applications."),
        ]
        embedder = OtariDocumentEmbedder(api_base_url=OTARI_API_BASE_URL, dimensions=256, progress_bar=False)
        result = embedder.run(docs)

        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert len(doc.embedding) == 256
            assert all(isinstance(x, float) for x in doc.embedding)
        assert "text-embedding-3-small" in result["meta"]["model"]
