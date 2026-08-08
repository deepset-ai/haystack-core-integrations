# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.rankers.cohere import CohereAzureRanker

AZURE_COHERE_API_URL = "https://my-endpoint.cohere.models.ai.azure.com"


class TestCohereAzureRanker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_COHERE_API_KEY", "test-api-key")
        component = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL)
        assert component.api_base_url == AZURE_COHERE_API_URL
        assert component.model == "rerank-v3.5"
        assert component.top_k == 10
        assert component.meta_fields_to_embed == []
        assert component.meta_data_separator == "\n"
        assert component.max_tokens_per_doc == 4096
        assert component.timeout == 30.0

    def test_init_with_parameters(self):
        component = CohereAzureRanker(
            api_base_url=AZURE_COHERE_API_URL,
            api_key=Secret.from_token("explicit-key"),
            model="rerank-v3.0",
            top_k=5,
            meta_fields_to_embed=["meta1", "meta2"],
            meta_data_separator=" | ",
            max_tokens_per_doc=2048,
            timeout=15.0,
        )
        assert component.api_base_url == AZURE_COHERE_API_URL
        assert component.api_key.resolve_value() == "explicit-key"
        assert component.model == "rerank-v3.0"
        assert component.top_k == 5
        assert component.meta_fields_to_embed == ["meta1", "meta2"]
        assert component.meta_data_separator == " | "
        assert component.max_tokens_per_doc == 2048
        assert component.timeout == 15.0

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_COHERE_API_KEY", "test-api-key")
        component = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL)
        dict_output = component.to_dict()
        assert dict_output == {
            "type": "haystack_integrations.components.rankers.cohere.azure_ranker.CohereAzureRanker",
            "init_parameters": {
                "api_base_url": AZURE_COHERE_API_URL,
                "api_key": {
                    "env_vars": ["COHERE_AZURE_API_KEY", "AZURE_COHERE_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "rerank-v3.5",
                "top_k": 10,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
                "max_tokens_per_doc": 4096,
                "timeout": 30.0,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_COHERE_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.rankers.cohere.azure_ranker.CohereAzureRanker",
            "init_parameters": {
                "api_base_url": AZURE_COHERE_API_URL,
                "api_key": {
                    "env_vars": ["COHERE_AZURE_API_KEY", "AZURE_COHERE_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "rerank-v3.5",
                "top_k": 5,
            },
        }
        component = CohereAzureRanker.from_dict(data)
        assert component.api_base_url == AZURE_COHERE_API_URL
        assert component.top_k == 5

    def test_run_empty_documents(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"))
        res = ranker.run(query="test", documents=[])
        assert res == {"documents": []}

    @pytest.mark.asyncio
    async def test_run_async_empty_documents(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"))
        res = await ranker.run_async(query="test", documents=[])
        assert res == {"documents": []}

    def test_run_invalid_top_k(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"), top_k=-1)
        docs = [Document(content="doc1")]
        with pytest.raises(ValueError, match="top_k must be > 0"):
            ranker.run(query="test", documents=docs)

    def test_run_success(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"), top_k=2)
        docs = [
            Document(content="Berlin"),
            Document(content="Paris"),
            Document(content="London"),
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
            ]
        }

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            output = ranker.run(query="capital of france", documents=docs)

            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            call_json = mock_post.call_args[1]["json"]
            call_headers = mock_post.call_args[1]["headers"]

            assert call_url == f"{AZURE_COHERE_API_URL}/v1/rerank"
            assert call_headers["api-key"] == "test-key"
            assert call_json["query"] == "capital of france"
            assert call_json["documents"] == ["Berlin", "Paris", "London"]
            assert call_json["top_n"] == 2

            res_docs = output["documents"]
            assert len(res_docs) == 2
            assert res_docs[0].content == "Paris"
            assert res_docs[0].score == 0.95
            assert res_docs[1].content == "Berlin"
            assert res_docs[1].score == 0.80

    @pytest.mark.asyncio
    async def test_run_async_success(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"), top_k=2)
        docs = [
            Document(content="Berlin"),
            Document(content="Paris"),
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.99},
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            output = await ranker.run_async(query="capital of france", documents=docs)

            mock_post.assert_called_once()
            res_docs = output["documents"]
            assert len(res_docs) == 1
            assert res_docs[0].content == "Paris"
            assert res_docs[0].score == 0.99

    def test_run_http_error(self):
        ranker = CohereAzureRanker(api_base_url=AZURE_COHERE_API_URL, api_key=Secret.from_token("test-key"))
        docs = [Document(content="doc1")]

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                ranker.run(query="test", documents=docs)
