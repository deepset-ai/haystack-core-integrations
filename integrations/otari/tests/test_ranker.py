# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.rankers.otari import OtariRanker

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "cohere:rerank-v3.5"
DEFAULT_API_BASE_URL = "https://api.otari.ai/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.rankers.otari.ranker.OtariRanker"


def _rerank_response(results: list[dict], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"id": "rerank-fake", "results": results, "meta": None, "usage": {"total_tokens": 10}},
    )


@pytest.fixture
def docs():
    return [
        Document(content="The capital of Brazil is Brasilia."),
        Document(content="The capital of France is Paris."),
    ]


SYNC_HELPER = "haystack_integrations.components.rankers.otari.ranker.request_with_retry"
ASYNC_HELPER = "haystack_integrations.components.rankers.otari.ranker.async_request_with_retry"


class TestInitializationAndSerialization:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        ranker = OtariRanker()
        assert ranker.model == DEFAULT_MODEL
        assert ranker.api_key == Secret.from_env_var("OTARI_API_KEY")
        assert ranker.api_base_url == DEFAULT_API_BASE_URL
        assert ranker.top_k is None
        assert ranker.score_threshold is None
        assert ranker.meta_fields_to_embed == []
        assert ranker.meta_data_separator == "\n"
        assert ranker.max_tokens_per_doc is None
        assert ranker.timeout == 30.0
        assert ranker.max_retries == 3

    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k must be > 0"):
            OtariRanker(top_k=0)

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        assert component_to_dict(OtariRanker(), "ranker") == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "model": DEFAULT_MODEL,
                "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": DEFAULT_API_BASE_URL,
                "top_k": None,
                "score_threshold": None,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
                "max_tokens_per_doc": None,
                "timeout": 30.0,
                "max_retries": 3,
            },
        }

    def test_to_dict_from_dict_round_trip_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        ranker = OtariRanker(
            model="voyage:rerank-2.5",
            api_key=Secret.from_env_var("ENV_VAR"),
            api_base_url="https://otari.example.com/api/v1",
            top_k=3,
            score_threshold=0.2,
            meta_fields_to_embed=["title"],
            meta_data_separator=" | ",
            max_tokens_per_doc=512,
            timeout=5.0,
            max_retries=1,
        )
        restored = component_from_dict(OtariRanker, component_to_dict(ranker, "ranker"), "ranker")
        assert restored.model == "voyage:rerank-2.5"
        assert restored.api_key == Secret.from_env_var("ENV_VAR")
        assert restored.api_base_url == "https://otari.example.com/api/v1"
        assert restored.top_k == 3
        assert restored.score_threshold == 0.2
        assert restored.meta_fields_to_embed == ["title"]
        assert restored.meta_data_separator == " | "
        assert restored.max_tokens_per_doc == 512
        assert restored.timeout == 5.0
        assert restored.max_retries == 1


class TestRequests:
    def test_key_resolved_at_run_not_init(self, monkeypatch, docs):
        monkeypatch.delenv("OTARI_API_KEY", raising=False)
        ranker = OtariRanker()

        with pytest.raises(ValueError, match="OTARI_API_KEY"):
            ranker.run(query="q", documents=docs)

    def test_request_uses_timeout_retries_and_headers(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"), timeout=5.0, max_retries=2)
        with patch(SYNC_HELPER, return_value=_rerank_response([{"index": 0, "relevance_score": 0.5}])) as helper:
            ranker.run(query="q", documents=docs)

        kwargs = helper.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "https://api.otari.ai/api/v1/rerank"
        assert kwargs["timeout"] == 5.0
        # max_retries counts the retries after the first attempt
        assert kwargs["attempts"] == 3
        assert kwargs["headers"] == {"Content-Type": "application/json", "Authorization": "Bearer test-api-key"}

    def test_http_status_error_raises_runtime_error_with_detail(self, docs):
        request = httpx.Request("POST", "https://api.otari.ai/api/v1/rerank")
        response = httpx.Response(401, json={"detail": "Invalid API key"}, request=request)
        error = httpx.HTTPStatusError("401", request=request, response=response)
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        with patch(SYNC_HELPER, side_effect=error), pytest.raises(RuntimeError, match="401: Invalid API key"):
            ranker.run(query="q", documents=docs)

    @pytest.mark.asyncio
    async def test_http_status_error_raises_runtime_error_with_detail_async(self, docs):
        request = httpx.Request("POST", "https://api.otari.ai/api/v1/rerank")
        response = httpx.Response(429, json={"detail": "Rate limit exceeded"}, request=request)
        error = httpx.HTTPStatusError("429", request=request, response=response)
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        with (
            patch(ASYNC_HELPER, new_callable=AsyncMock, side_effect=error),
            pytest.raises(RuntimeError, match="429: Rate limit exceeded"),
        ):
            await ranker.run_async(query="q", documents=docs)

    def test_warns_once_about_eu_key_on_default_url(self, caplog, docs):
        ranker = OtariRanker(api_key=Secret.from_token("otk_v1_eu_test"))
        with patch(SYNC_HELPER, return_value=_rerank_response([{"index": 0, "relevance_score": 0.5}])):
            ranker.run(query="q", documents=docs)
            ranker.run(query="q", documents=docs)
        assert caplog.text.count("belongs to otari.ai's EU region") == 1

    @pytest.mark.asyncio
    async def test_warns_once_about_eu_key_on_default_url_async(self, caplog, docs):
        ranker = OtariRanker(api_key=Secret.from_token("otk_v1_eu_test"))
        with patch(
            ASYNC_HELPER, new_callable=AsyncMock, return_value=_rerank_response([{"index": 0, "relevance_score": 0.5}])
        ):
            await ranker.run_async(query="q", documents=docs)
            await ranker.run_async(query="q", documents=docs)
        assert caplog.text.count("belongs to otari.ai's EU region") == 1


class TestRun:
    def test_prepare_texts_with_meta(self):
        ranker = OtariRanker(
            api_key=Secret.from_token("test-api-key"), meta_fields_to_embed=["topic"], meta_data_separator=" | "
        )
        documents = [Document(content="hello", meta={"topic": "ML"}), Document(content="world", meta={})]
        assert ranker._prepare_texts(documents) == ["ML | hello", "world"]

    def test_prepare_request(self):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"), max_tokens_per_doc=512)
        body = ranker._prepare_request(query="q", documents=[Document(content="a")], top_k=2)
        assert body == {"model": DEFAULT_MODEL, "query": "q", "documents": ["a"], "top_n": 2, "max_tokens_per_doc": 512}

    def test_prepare_request_without_optional_fields(self):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        body = ranker._prepare_request(query="q", documents=[Document(content="a")], top_k=None)
        assert body == {"model": DEFAULT_MODEL, "query": "q", "documents": ["a"]}

    def test_parse_response_applies_score_threshold(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        response = _rerank_response([{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}])
        out = ranker._parse_response(response, docs, score_threshold=0.5)
        assert [doc.content for doc in out["documents"]] == ["The capital of France is Paris."]
        assert out["documents"][0].score == 0.9
        assert out["meta"] == {"model": DEFAULT_MODEL, "usage": {"total_tokens": 10}}

    def test_parse_response_reports_the_model_the_gateway_names(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"), model="fast-rerank")
        response = httpx.Response(200, json={"model": "rerank-v3.5", "results": [{"index": 1, "relevance_score": 0.9}]})
        out = ranker._parse_response(response, docs, score_threshold=None)
        assert out["meta"] == {"model": "rerank-v3.5", "usage": {}}

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (httpx.Response(401, json={"detail": "Invalid API key"}), "status code 401: Invalid API key"),
            (
                httpx.Response(422, json={"detail": [{"loc": ["body", "top_n"], "msg": "Input should be > 0"}]}),
                "status code 422: .*Input should be > 0",
            ),
            (httpx.Response(502, text="Bad Gateway"), "status code 502: Bad Gateway"),
            (httpx.Response(200, json={"unexpected": "body"}), "status code 200"),
        ],
    )
    def test_parse_response_raises_on_error(self, response, message):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(RuntimeError, match=message):
            ranker._parse_response(response, [], score_threshold=None)

    def test_run_invalid_top_k(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        with pytest.raises(ValueError, match="top_k must be > 0"):
            ranker.run(query="q", documents=docs, top_k=0)

    def test_run_empty_documents(self):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        assert ranker.run(query="q", documents=[]) == {"documents": [], "meta": {}}

    def test_run(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"), top_k=2)
        response = _rerank_response([{"index": 1, "relevance_score": 0.99}, {"index": 0, "relevance_score": 0.01}])
        with patch(SYNC_HELPER, return_value=response) as helper:
            out = ranker.run(query="What is the capital of France?", documents=docs)

        helper.assert_called_once()
        assert helper.call_args.kwargs["json"] == {
            "model": DEFAULT_MODEL,
            "query": "What is the capital of France?",
            "documents": [doc.content for doc in docs],
            "top_n": 2,
        }
        assert [doc.content for doc in out["documents"]] == [
            "The capital of France is Paris.",
            "The capital of Brazil is Brasilia.",
        ]
        assert out["documents"][0].score == 0.99

    @pytest.mark.asyncio
    async def test_run_async(self, docs):
        ranker = OtariRanker(api_key=Secret.from_token("test-api-key"))
        response = _rerank_response([{"index": 1, "relevance_score": 0.42}])
        with patch(ASYNC_HELPER, new_callable=AsyncMock, return_value=response) as helper:
            out = await ranker.run_async(query="q", documents=docs, top_k=1)

        assert helper.call_args.kwargs["json"]["top_n"] == 1
        assert [doc.content for doc in out["documents"]] == ["The capital of France is Paris."]
        assert out["documents"][0].score == 0.42


class TestIntegration:
    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self, docs):
        ranker = OtariRanker(api_base_url=OTARI_API_BASE_URL)
        out = ranker.run(query="What is the capital of France?", documents=docs)

        assert out["documents"][0].content == "The capital of France is Paris."
        assert out["documents"][0].score > out["documents"][1].score

    @requires_api_key
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self, docs):
        ranker = OtariRanker(api_base_url=OTARI_API_BASE_URL, top_k=1)
        out = await ranker.run_async(query="What is the capital of France?", documents=docs)

        assert len(out["documents"]) == 1
        assert out["documents"][0].content == "The capital of France is Paris."
