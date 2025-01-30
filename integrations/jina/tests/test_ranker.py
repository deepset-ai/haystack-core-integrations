# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch

import pytest
import requests
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.jina import JinaRanker


def mock_session_post_response(*args, **kwargs):  # noqa: ARG001
    model = kwargs["json"]["model"]
    documents = kwargs["json"]["documents"]
    mock_response = requests.Response()
    mock_response.status_code = 200
    results = [
        {"index": i, "relevance_score": len(documents) - i, "document": {"text": doc}}
        for i, doc in enumerate(documents)
    ]
    mock_response._content = json.dumps(
        {"model": model, "usage": {"total_tokens": 4, "prompt_tokens": 4}, "results": results}
    ).encode()

    return mock_response


class TestJinaRanker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaRanker()

        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")
        assert embedder.model == "jina-reranker-v1-base-en"

    def test_init_with_parameters(self):
        embedder = JinaRanker(api_key=Secret.from_token("fake-api-key"), model="model", top_k=64, score_threshold=0.5)

        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == "model"
        assert embedder.top_k == 64
        assert embedder.score_threshold == 0.5

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JinaRanker()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.jina.ranker.JinaRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-reranker-v1-base-en",
                "top_k": None,
                "score_threshold": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaRanker(model="model", top_k=64, score_threshold=0.5)
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.jina.ranker.JinaRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "model",
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
        with patch("requests.sessions.Session.post", side_effect=mock_session_post_response):
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
