# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import httpx
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.jina import JinaRanker


def mock_httpx_post_response(*args, **kwargs):  # noqa: ARG001
    model = kwargs["json"]["model"]
    documents = kwargs["json"]["documents"]
    results = [
        {"index": i, "relevance_score": len(documents) - i, "document": {"text": doc}}
        for i, doc in enumerate(documents)
    ]
    return httpx.Response(
        200,
        json={"model": model, "usage": {"total_tokens": 4, "prompt_tokens": 4}, "results": results},
    )


class TestJinaRanker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaRanker()

        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")
        assert embedder.model == "jina-reranker-v1-base-en"

    def test_init_with_parameters(self):
        embedder = JinaRanker(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            base_url="https://my.custom.url/v1/rerank",
            top_k=64,
            score_threshold=0.5,
        )

        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == "model"
        assert embedder.base_url == "https://my.custom.url/v1/rerank"
        assert embedder.top_k == 64
        assert embedder.score_threshold == 0.5

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JinaRanker()

    def test_init_fails_with_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
            JinaRanker(api_key=Secret.from_token("fake-api-key"), top_k=0)

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.rankers.jina.ranker.JinaRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
                "base_url": "https://my.custom.url/v1/rerank",
                "top_k": 5,
                "score_threshold": 0.3,
            },
        }
        ranker = JinaRanker.from_dict(data)

        assert ranker.api_key == Secret.from_env_var("JINA_API_KEY")
        assert ranker.model == "model"
        assert ranker.base_url == "https://my.custom.url/v1/rerank"
        assert ranker.top_k == 5
        assert ranker.score_threshold == 0.3

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.jina.ranker.JinaRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-reranker-v1-base-en",
                "base_url": "https://api.jina.ai/v1/rerank",
                "top_k": None,
                "score_threshold": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaRanker(model="model", top_k=64, score_threshold=0.5, base_url="https://my.custom.url/v1/rerank")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.jina.ranker.JinaRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
                "base_url": "https://my.custom.url/v1/rerank",
                "top_k": 64,
                "score_threshold": 0.5,
            },
        }

    def test_run(self):
        docs = [
            Document(content="I love cheese"),
            Document(content="A transformer is a deep learning architecture"),
            Document(content="A transformer is something"),
            Document(content="A transformer is not good"),
        ]
        query = "What is a transformer?"

        model = "jina-ranker"
        with patch("httpx.Client.post", side_effect=mock_httpx_post_response):
            ranker = JinaRanker(
                api_key=Secret.from_token("fake-api-key"),
                model=model,
            )

            result = ranker.run(query=query, documents=docs)

        ranked_documents = result["documents"]
        metadata = result["meta"]

        assert isinstance(ranked_documents, list)
        assert len(ranked_documents) == len(docs)
        for i, doc in enumerate(ranked_documents):
            assert isinstance(doc, Document)
            assert doc.score == len(ranked_documents) - i
        assert metadata == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    def test_run_does_not_modify_original_documents(self):
        docs = [
            Document(content="I love cheese"),
            Document(content="A transformer is a deep learning architecture"),
            Document(content="A transformer is something"),
            Document(content="A transformer is not good"),
        ]
        query = "What is a transformer?"

        model = "jina-ranker"
        with patch("httpx.Client.post", side_effect=mock_httpx_post_response):
            ranker = JinaRanker(
                api_key=Secret.from_token("fake-api-key"),
                model=model,
            )

            result = ranker.run(query=query, documents=docs)

        # originals remain unchanged
        for doc in docs:
            assert doc.score is None

        # returned docs carry scores
        for doc in result["documents"]:
            assert doc.score is not None

    @pytest.mark.asyncio
    async def test_run_async(self):
        docs = [
            Document(content="I love cheese"),
            Document(content="A transformer is a deep learning architecture"),
            Document(content="A transformer is something"),
            Document(content="A transformer is not good"),
        ]
        query = "What is a transformer?"

        model = "jina-ranker"
        with patch("httpx.AsyncClient.post", side_effect=mock_httpx_post_response):
            ranker = JinaRanker(
                api_key=Secret.from_token("fake-api-key"),
                model=model,
            )

            result = await ranker.run_async(query=query, documents=docs)

        ranked_documents = result["documents"]
        metadata = result["meta"]

        assert isinstance(ranked_documents, list)
        assert len(ranked_documents) == len(docs)
        for i, doc in enumerate(ranked_documents):
            assert isinstance(doc, Document)
            assert doc.score == len(ranked_documents) - i
        assert metadata == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    @pytest.mark.asyncio
    async def test_run_async_on_empty_docs(self):
        ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"))

        result = await ranker.run_async(query="a", documents=[])

        assert result["documents"] is not None
        assert not result["documents"]

    def test_run_wrong_input_format(self):
        ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
            ranker.run(query="query", documents=[Document(content="document")], top_k=0)

    def test_run_on_empty_docs(self):
        ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"))

        empty_list_input = []
        result = ranker.run(query="a", documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    def test_run_raises_runtime_error_on_api_error(self):
        mock_response = httpx.Response(400, json={"detail": "Bad request"})
        with patch("httpx.Client.post", return_value=mock_response):
            ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"))
            with pytest.raises(RuntimeError, match="Bad request"):
                ranker.run(query="q", documents=[Document(content="doc")])

    def test_run_with_score_threshold_filters_results(self):
        docs = [Document(content=f"doc {i}") for i in range(4)]

        with patch("httpx.Client.post", side_effect=mock_httpx_post_response):
            ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"), score_threshold=2.5)
            result = ranker.run(query="q", documents=docs)

        # mock assigns scores len(docs)-i, so for 4 docs scores are 4, 3, 2, 1 - only first two pass
        ranked = result["documents"]
        assert len(ranked) == 2
        assert all(doc.score >= 2.5 for doc in ranked)

    def test_run_with_top_k_truncates_results(self):
        docs = [Document(content=f"doc {i}") for i in range(5)]

        with patch("httpx.Client.post", side_effect=mock_httpx_post_response):
            ranker = JinaRanker(api_key=Secret.from_token("fake-api-key"))
            result = ranker.run(query="q", documents=docs, top_k=2)

        assert len(result["documents"]) == 2

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    def test_run_integration(self):
        ranker = JinaRanker(model="jina-reranker-v1-base-en")
        docs = [
            Document(content="Paris is the capital of France."),
            Document(content="Bananas are yellow fruits."),
            Document(content="Berlin is the capital of Germany."),
        ]

        result = ranker.run(query="What is the capital of France?", documents=docs, top_k=2)

        assert "documents" in result
        ranked_docs = result["documents"]
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)
        assert all(doc.score is not None for doc in ranked_docs)
        assert ranked_docs[0].score >= ranked_docs[1].score
        assert "Paris" in ranked_docs[0].content

        assert "meta" in result
        assert isinstance(result["meta"], dict)
        assert "model" in result["meta"]
        assert "usage" in result["meta"]

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        ranker = JinaRanker(model="jina-reranker-v1-base-en")
        docs = [
            Document(content="Paris is the capital of France."),
            Document(content="Bananas are yellow fruits."),
            Document(content="Berlin is the capital of Germany."),
        ]

        result = await ranker.run_async(query="What is the capital of France?", documents=docs, top_k=2)

        assert "documents" in result
        ranked_docs = result["documents"]
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)
        assert all(doc.score is not None for doc in ranked_docs)
        assert ranked_docs[0].score >= ranked_docs[1].score
        assert "Paris" in ranked_docs[0].content

        assert "meta" in result
        assert isinstance(result["meta"], dict)
        assert "model" in result["meta"]
        assert "usage" in result["meta"]
