# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.rankers.vllm import VLLMRanker

MODEL = "BAAI/bge-reranker-base"
API_BASE_URL = "http://localhost:8002/v1"


def _fake_response(results: list[dict], model: str = "fake-model", total_tokens: int = 10):
    response = MagicMock()
    response.json.return_value = {
        "id": "rerank-fake",
        "model": model,
        "usage": {"total_tokens": total_tokens},
        "results": results,
    }
    return response


class TestInitializationAndSerialization:
    def test_init_default(self):
        ranker = VLLMRanker(model=MODEL)
        assert ranker.model == MODEL
        assert ranker.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert ranker.api_base_url == "http://localhost:8000/v1"
        assert ranker.top_k is None
        assert ranker.score_threshold is None
        assert ranker.meta_fields_to_embed == []
        assert ranker.meta_data_separator == "\n"
        assert ranker.http_client_kwargs is None
        assert ranker.extra_parameters is None
        assert ranker._client is None
        assert ranker._async_client is None

    def test_init_with_parameters(self):
        ranker = VLLMRanker(
            model=MODEL,
            api_key=Secret.from_token("test-api-key"),
            api_base_url="http://my-vllm-server:8000/v1",
            top_k=5,
            score_threshold=0.5,
            meta_fields_to_embed=["topic"],
            meta_data_separator=" | ",
            http_client_kwargs={"verify": False},
            extra_parameters={"truncate_prompt_tokens": 256},
        )
        assert ranker.api_key == Secret.from_token("test-api-key")
        assert ranker.api_base_url == "http://my-vllm-server:8000/v1"
        assert ranker.top_k == 5
        assert ranker.score_threshold == 0.5
        assert ranker.meta_fields_to_embed == ["topic"]
        assert ranker.meta_data_separator == " | "
        assert ranker.http_client_kwargs == {"verify": False}
        assert ranker.extra_parameters == {"truncate_prompt_tokens": 256}
        assert ranker._client is None
        assert ranker._async_client is None

    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k must be > 0"):
            VLLMRanker(model=MODEL, top_k=0)

    def test_to_dict(self):
        component_dict = component_to_dict(VLLMRanker(model=MODEL), "ranker")
        assert component_dict == {
            "type": "haystack_integrations.components.rankers.vllm.ranker.VLLMRanker",
            "init_parameters": {
                "model": MODEL,
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "api_base_url": "http://localhost:8000/v1",
                "top_k": None,
                "score_threshold": None,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
                "http_client_kwargs": None,
                "extra_parameters": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.rankers.vllm.ranker.VLLMRanker",
            "init_parameters": {
                "model": MODEL,
                "api_key": {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"},
                "api_base_url": "http://localhost:8000/v1",
                "top_k": 3,
                "score_threshold": 0.5,
                "meta_fields_to_embed": ["topic"],
                "meta_data_separator": " | ",
                "http_client_kwargs": None,
                "extra_parameters": {"truncate_prompt_tokens": 256},
            },
        }
        ranker = component_from_dict(VLLMRanker, data, "ranker")
        assert ranker.model == MODEL
        assert ranker.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert ranker.top_k == 3
        assert ranker.score_threshold == 0.5
        assert ranker.meta_fields_to_embed == ["topic"]
        assert ranker.meta_data_separator == " | "
        assert ranker.extra_parameters == {"truncate_prompt_tokens": 256}


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        ranker = VLLMRanker(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            ranker.warm_up()

    def test_sync_lifecycle(self):
        client = MagicMock(spec=httpx.Client)
        ranker = VLLMRanker(model=MODEL, api_key=Secret.from_token("test-api-key"))

        with patch(
            "haystack_integrations.components.rankers.vllm.ranker.init_http_client", return_value=client
        ) as mock_init:
            ranker.warm_up()
            assert ranker._client is client
            assert ranker._async_client is None
            assert mock_init.call_args.kwargs["http_client_kwargs"]["headers"]["Authorization"] == (
                "Bearer test-api-key"
            )
            ranker.close()
            client.close.assert_called_once_with()
            assert ranker._client is None
            ranker.warm_up()
            assert mock_init.call_count == 2

    @pytest.mark.asyncio
    async def test_async_lifecycle(self):
        client = MagicMock(spec=httpx.AsyncClient)
        client.aclose = AsyncMock()
        ranker = VLLMRanker(model=MODEL)

        with patch(
            "haystack_integrations.components.rankers.vllm.ranker.init_http_client", return_value=client
        ) as mock_init:
            await ranker.warm_up_async()
            assert ranker._async_client is client
            assert ranker._client is None
            await ranker.close_async()
            client.aclose.assert_awaited_once_with()
            assert ranker._async_client is None
            await ranker.warm_up_async()
            assert mock_init.call_count == 2

    def test_warm_up_is_idempotent(self):
        client = MagicMock(spec=httpx.Client)
        ranker = VLLMRanker(model=MODEL)
        with patch(
            "haystack_integrations.components.rankers.vllm.ranker.init_http_client", return_value=client
        ) as mock_init:
            ranker.warm_up()
            ranker.warm_up()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self):
        client = MagicMock(spec=httpx.AsyncClient)
        ranker = VLLMRanker(model=MODEL)
        with patch(
            "haystack_integrations.components.rankers.vllm.ranker.init_http_client", return_value=client
        ) as mock_init:
            await ranker.warm_up_async()
            await ranker.warm_up_async()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        ranker = VLLMRanker(model=MODEL)
        ranker.close()
        await ranker.close_async()
        assert ranker._client is None
        assert ranker._async_client is None

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        ranker = VLLMRanker(model=MODEL)
        sync_client = MagicMock()
        async_client = MagicMock(aclose=AsyncMock())
        ranker._client = sync_client
        ranker._async_client = async_client

        ranker.close()
        assert ranker._client is None
        assert ranker._async_client is async_client
        async_client.aclose.assert_not_awaited()
        await ranker.close_async()
        assert ranker._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_prepare_texts_with_meta(self):
        ranker = VLLMRanker(model=MODEL, meta_fields_to_embed=["topic"], meta_data_separator=" | ")
        docs = [Document(content="hello", meta={"topic": "ML"}), Document(content="world", meta={})]
        assert ranker._prepare_texts(docs) == ["ML | hello", "world"]

    def test_prepare_request_with_top_k_and_extras(self):
        ranker = VLLMRanker(model=MODEL, extra_parameters={"truncate_prompt_tokens": 256})
        docs = [Document(content="a"), Document(content="b")]
        body = ranker._prepare_request(query="q", documents=docs, top_k=2)
        assert body == {
            "model": MODEL,
            "query": "q",
            "documents": ["a", "b"],
            "top_n": 2,
            "truncate_prompt_tokens": 256,
        }

    def test_prepare_request_without_top_k(self):
        ranker = VLLMRanker(model=MODEL)
        body = ranker._prepare_request(query="q", documents=[Document(content="a")], top_k=None)
        assert "top_n" not in body

    def test_parse_response_applies_score_threshold(self):
        docs = [Document(content="a"), Document(content="b")]
        resp = {
            "model": "fake-model",
            "usage": {"total_tokens": 5},
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ],
        }
        out = VLLMRanker._parse_response(resp, docs, score_threshold=0.5)
        assert [d.content for d in out["documents"]] == ["b"]
        assert out["documents"][0].score == 0.9
        assert out["meta"] == {"model": "fake-model", "usage": {"total_tokens": 5}}

    def test_parse_response_raises_on_error(self):
        with pytest.raises(RuntimeError, match="boom"):
            VLLMRanker._parse_response({"detail": "boom"}, [], score_threshold=None)

    def test_run_invalid_top_k(self):
        ranker = VLLMRanker(model=MODEL)
        with pytest.raises(ValueError, match="top_k must be > 0"):
            ranker.run(query="q", documents=[Document(content="a")], top_k=0)

    def test_run_empty_documents(self):
        ranker = VLLMRanker(model=MODEL)
        assert ranker.run(query="q", documents=[]) == {"documents": [], "meta": {}}

    def test_run(self):
        ranker = VLLMRanker(model=MODEL, top_k=2)
        ranker._client = MagicMock()
        ranker._client.post.return_value = _fake_response(
            results=[
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.01},
            ]
        )

        docs = [
            Document(content="The capital of Brazil is Brasilia."),
            Document(content="The capital of France is Paris."),
        ]
        out = ranker.run(query="What is the capital of France?", documents=docs)

        ranker._client.post.assert_called_once()
        call_kwargs = ranker._client.post.call_args
        assert call_kwargs.args[0] == "http://localhost:8000/v1/rerank"
        assert call_kwargs.kwargs["json"] == {
            "model": MODEL,
            "query": "What is the capital of France?",
            "documents": [d.content for d in docs],
            "top_n": 2,
        }
        assert [d.content for d in out["documents"]] == [
            "The capital of France is Paris.",
            "The capital of Brazil is Brasilia.",
        ]
        assert out["documents"][0].score == 0.99
        assert out["meta"]["model"] == "fake-model"

    @pytest.mark.asyncio
    async def test_run_async(self):
        ranker = VLLMRanker(model=MODEL)
        ranker._async_client = MagicMock()
        ranker._async_client.post = AsyncMock(
            return_value=_fake_response(results=[{"index": 0, "relevance_score": 0.42}])
        )

        out = await ranker.run_async(query="q", documents=[Document(content="a")])

        assert out["documents"][0].score == 0.42


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self):
        ranker = VLLMRanker(model=MODEL, api_base_url=API_BASE_URL)
        docs = [
            Document(content="The capital of Brazil is Brasilia."),
            Document(content="The capital of France is Paris."),
        ]
        out = ranker.run(query="What is the capital of France?", documents=docs)
        assert out["documents"][0].content == "The capital of France is Paris."
        assert out["documents"][0].score is not None
        assert out["meta"]["model"] == MODEL

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        ranker = VLLMRanker(model=MODEL, api_base_url=API_BASE_URL)
        docs = [
            Document(content="The capital of Brazil is Brasilia."),
            Document(content="The capital of France is Paris."),
        ]
        out = await ranker.run_async(query="What is the capital of France?", documents=docs)
        assert out["documents"][0].content == "The capital of France is Paris."
