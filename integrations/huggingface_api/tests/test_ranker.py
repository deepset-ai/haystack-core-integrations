# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.huggingface_api import HuggingFaceTEIRanker, TruncationDirection


class TestHuggingFaceTEIRanker:
    def test_init(self, del_hf_env_vars_if_empty):
        """Test initialization with default and custom parameters"""
        # Default parameters
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        assert ranker.url == "https://api.my-tei-service.com"
        assert ranker.top_k == 10
        assert ranker.timeout == 30
        assert ranker.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert ranker.max_retries == 3
        assert ranker.retry_status_codes is None

        # Custom parameters
        token = Secret.from_token("my_api_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=5,
            timeout=60,
            token=token,
            max_retries=5,
            retry_status_codes=[500, 502, 503],
        )
        assert ranker.url == "https://api.my-tei-service.com"
        assert ranker.top_k == 5
        assert ranker.timeout == 60
        assert ranker.token == token
        assert ranker.max_retries == 5
        assert ranker.retry_status_codes == [500, 502, 503]

    def test_to_dict(self, del_hf_env_vars_if_empty):
        """Test serialization to dict with Secret token"""
        component = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com", top_k=5, timeout=30, max_retries=4, retry_status_codes=[500, 502]
        )
        data = component.to_dict()

        assert data["type"] == "haystack_integrations.components.rankers.huggingface_api.ranker.HuggingFaceTEIRanker"
        assert data["init_parameters"]["url"] == "https://api.my-tei-service.com"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["timeout"] == 30
        assert data["init_parameters"]["token"] == {
            "env_vars": ["HF_API_TOKEN", "HF_TOKEN"],
            "strict": False,
            "type": "env_var",
        }
        assert data["init_parameters"]["max_retries"] == 4
        assert data["init_parameters"]["retry_status_codes"] == [500, 502]

    @pytest.mark.parametrize("use_grpc", [False, True])
    def test_raw_scores_survives_a_serialization_round_trip(self, del_hf_env_vars_if_empty, use_grpc):
        """raw_scores goes into the request payload, so losing it changes what the API returns."""
        ranker = HuggingFaceTEIRanker(
            url="localhost:8081" if use_grpc else "https://api.my-tei-service.com",
            raw_scores=True,
            use_grpc=use_grpc,
        )

        restored = HuggingFaceTEIRanker.from_dict(ranker.to_dict())

        assert restored.raw_scores is True
        assert restored.use_grpc is use_grpc
        assert restored.url == ranker.url

    def test_from_dict(self, del_hf_env_vars_if_empty):
        """Test deserialization from dict with environment variable token"""
        data = {
            "type": "haystack_integrations.components.rankers.huggingface_api.ranker.HuggingFaceTEIRanker",
            "init_parameters": {
                "url": "https://api.my-tei-service.com",
                "top_k": 5,
                "timeout": 30,
                "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                "max_retries": 4,
                "retry_status_codes": [500, 502],
            },
        }

        component = HuggingFaceTEIRanker.from_dict(data)

        assert component.url == "https://api.my-tei-service.com"
        assert component.top_k == 5
        assert component.timeout == 30
        assert component.max_retries == 4
        assert component.retry_status_codes == [500, 502]

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.Client")
    def test_grpc_lifecycle(self, mock_client):
        ranker = HuggingFaceTEIRanker(url="localhost:8081", use_grpc=True)
        mock_client.assert_not_called()

        ranker.warm_up()
        ranker.warm_up()
        mock_client.assert_called_once_with("localhost:8081")

        ranker.close()
        ranker.close()
        mock_client.return_value.channel.close.assert_called_once()
        assert ranker._grpc_client is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.AsyncClient.create")
    async def test_grpc_lifecycle_async(self, mock_create):
        mock_create.return_value.channel.close = AsyncMock()
        ranker = HuggingFaceTEIRanker(url="localhost:8081", use_grpc=True)
        mock_create.assert_not_called()

        await ranker.warm_up_async()
        await ranker.warm_up_async()
        mock_create.assert_awaited_once_with("localhost:8081")

        await ranker.close_async()
        await ranker.close_async()
        mock_create.return_value.channel.close.assert_awaited_once()
        assert ranker._async_grpc_client is None

    def test_empty_documents(self, del_hf_env_vars_if_empty):
        """Test that empty documents list returns empty result"""
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        result = ranker.run(query="test query", documents=[])
        assert result == {"documents": []}

    def test_init_fail_with_zero_top_k(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
            HuggingFaceTEIRanker(url="http://localhost:8080", top_k=0)

    def test_init_fail_with_negative_top_k(self):
        with pytest.raises(ValueError, match=r"top_k must be > 0, but got -5"):
            HuggingFaceTEIRanker(url="http://localhost:8080", top_k=-5)

    def test_run_fail_with_invalid_top_k_override(self):
        ranker = HuggingFaceTEIRanker(url="http://localhost:8080")
        docs = [Document(content="doc")]
        with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
            ranker.run(query="q", documents=docs, top_k=0)

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.request_with_retry")
    def test_run_with_mock(self, mock_request, del_hf_env_vars_if_empty):
        """Test run method with mocked API response"""
        # Setup mock response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"index": 2, "score": 0.95},
            {"index": 1, "score": 0.85},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=3,
            timeout=30,
            token=token,
            max_retries=4,
            retry_status_codes=[500, 502],
        )

        docs = [Document(content="Document A"), Document(content="Document B"), Document(content="Document C")]

        # Run the ranker
        result = ranker.run(query="test query", documents=docs)

        # Check that request_with_retry was called with correct parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["Document A", "Document B", "Document C"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=4,
            status_codes_to_retry=[500, 502],
        )

        # Check that documents are ranked correctly
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document C"
        assert result["documents"][0].score == 0.95
        assert result["documents"][1].content == "Document B"
        assert result["documents"][1].score == 0.85
        assert result["documents"][2].content == "Document A"
        assert result["documents"][2].score == 0.75

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.Client")
    def test_run_grpc(self, mock_client):
        client = mock_client.return_value
        client.unary_unary.return_value = {"ranks": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]}
        ranker = HuggingFaceTEIRanker(
            url="localhost:8081",
            use_grpc=True,
            raw_scores=True,
            timeout=12,
            token=Secret.from_token("grpc-test-key"),
        )
        documents = [Document(content="Bananas are yellow."), Document(content="Paris is in France.")]

        result = ranker.run(query="Where is Paris?", documents=documents)

        client.unary_unary.assert_called_once_with(
            "tei.v1.Rerank",
            "Rerank",
            {"query": "Where is Paris?", "texts": [doc.content for doc in documents], "raw_scores": True},
            timeout=12,
            metadata=(("authorization", "Bearer grpc-test-key"),),
        )
        assert result["documents"] == [replace(documents[1], score=0.9), replace(documents[0], score=0.2)]

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            (TruncationDirection.LEFT, "TRUNCATION_DIRECTION_LEFT"),
            (TruncationDirection.RIGHT, "TRUNCATION_DIRECTION_RIGHT"),
        ],
    )
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.Client")
    def test_run_grpc_with_truncation_direction(self, mock_client, direction, expected):
        client = mock_client.return_value
        client.unary_unary.return_value = {"ranks": [{"index": 0, "score": 0.9}]}
        ranker = HuggingFaceTEIRanker(url="localhost:8081", use_grpc=True, token=None)

        ranker.run(query="query", documents=[Document(content="document")], truncation_direction=direction)

        payload = client.unary_unary.call_args.args[2]
        assert payload["truncate"] is True
        assert payload["truncation_direction"] == expected

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.request_with_retry")
    def test_run_with_truncation_direction(self, mock_request, del_hf_env_vars_if_empty):
        """Test run method with truncation direction parameter"""
        # Setup mock response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [{"index": 0, "score": 0.95}]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com", token=token)
        docs = [Document(content="Document A")]

        # Run the ranker with truncation direction
        ranker.run(query="test query", documents=docs, truncation_direction=TruncationDirection.LEFT)

        # Check that request includes truncation parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={
                "query": "test query",
                "texts": ["Document A"],
                "raw_scores": False,
                "truncate": True,
                "truncation_direction": "Left",
            },
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=3,
            status_codes_to_retry=None,
        )

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.request_with_retry")
    def test_run_with_custom_top_k(self, mock_request, del_hf_env_vars_if_empty):
        """Test run method with custom top_k parameter"""
        # Setup mock response with 5 documents
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"index": 4, "score": 0.95},
            {"index": 3, "score": 0.90},
            {"index": 2, "score": 0.85},
            {"index": 1, "score": 0.80},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker with top_k=3
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com", top_k=3)

        # Create 5 test documents
        docs = [Document(content=f"Document {i}") for i in range(5)]

        # Run the ranker
        result = ranker.run(query="test query", documents=docs)

        # Check that only top 3 documents are returned
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document 4"
        assert result["documents"][1].content == "Document 3"
        assert result["documents"][2].content == "Document 2"

        # Test with run-time top_k override
        result = ranker.run(query="test query", documents=docs, top_k=2)

        # Check that only top 2 documents are returned
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Document 4"
        assert result["documents"][1].content == "Document 3"

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.request_with_retry")
    def test_run_deduplicates_documents(self, mock_request, del_hf_env_vars_if_empty):
        """Test that duplicate documents are removed before sending to the API."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]
        mock_request.return_value = mock_response

        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        # Document with duplicate id and lower score should be dropped
        docs = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique"),
        ]

        result = ranker.run(query="test query", documents=docs)

        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["keep me", "unique"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": f"Bearer {ranker.token.resolve_value()}"} if ranker.token.resolve_value() else {},
            attempts=3,
            status_codes_to_retry=None,
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "unique"
        assert result["documents"][1].content == "keep me"

    @pytest.mark.parametrize(
        ("rank", "score"),
        [({"score": 0.9}, 0.9), ({"index": 0}, 0.0), ({}, 0.0)],
    )
    def test_compose_grpc_response_restores_defaults(self, rank, score):
        ranker = HuggingFaceTEIRanker(url="localhost:8081", use_grpc=True)
        document = Document(content="document")

        result = ranker._compose_grpc_response({"ranks": [rank]}, top_k=None, documents=[document])

        assert result == {"documents": [replace(document, score=score)]}

    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.request_with_retry")
    def test_error_handling(self, mock_request, del_hf_env_vars_if_empty):
        """Test error handling in the ranker"""
        # Setup mock response with error
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": "Some error occurred", "error_type": "TestError"}
        mock_request.return_value = mock_response

        # Create ranker and test documents
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        docs = [Document(content="Document A")]

        # Test that RuntimeError is raised with the correct message
        with pytest.raises(
            RuntimeError, match=r"HuggingFaceTEIRanker API call failed \(TestError\): Some error occurred"
        ):
            ranker.run(query="test query", documents=docs)

        # Test unexpected response format
        mock_response.json.return_value = {"unexpected": "format"}
        with pytest.raises(TypeError, match="Unexpected response format from text-embeddings-inference rerank API"):
            ranker.run(query="test query", documents=docs)

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.async_request_with_retry")
    async def test_run_async_with_mock(self, mock_request, del_hf_env_vars_if_empty):
        """Test run_async method with mocked API response"""
        # Setup mock response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"index": 2, "score": 0.95},
            {"index": 1, "score": 0.85},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=3,
            timeout=30,
            token=token,
            max_retries=4,
            retry_status_codes=[500, 502],
        )

        docs = [Document(content="Document A"), Document(content="Document B"), Document(content="Document C")]

        # Run the ranker asynchronously
        result = await ranker.run_async(query="test query", documents=docs)

        # Check that async_request_with_retry was called with correct parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["Document A", "Document B", "Document C"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=4,
            status_codes_to_retry=[500, 502],
        )

        # Check that documents are ranked correctly
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document C"
        assert result["documents"][0].score == 0.95
        assert result["documents"][1].content == "Document B"
        assert result["documents"][1].score == 0.85
        assert result["documents"][2].content == "Document A"
        assert result["documents"][2].score == 0.75

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.AsyncClient.create")
    async def test_run_async_grpc(self, mock_client):
        client = mock_client.return_value
        client.unary_unary.return_value = {"ranks": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]}
        ranker = HuggingFaceTEIRanker(
            url="localhost:8081",
            use_grpc=True,
            raw_scores=True,
            timeout=12,
            token=Secret.from_token("grpc-test-key"),
        )
        documents = [Document(content="Bananas are yellow."), Document(content="Paris is in France.")]

        result = await ranker.run_async(query="Where is Paris?", documents=documents)

        client.unary_unary.assert_awaited_once_with(
            "tei.v1.Rerank",
            "Rerank",
            {"query": "Where is Paris?", "texts": [doc.content for doc in documents], "raw_scores": True},
            timeout=12,
            metadata=(("authorization", "Bearer grpc-test-key"),),
        )
        assert result["documents"] == [replace(documents[1], score=0.9), replace(documents[0], score=0.2)]

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.async_request_with_retry")
    async def test_run_async_deduplicates_documents(self, mock_request, del_hf_env_vars_if_empty):
        """Test that duplicate documents are removed before sending to the API."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]
        mock_request.return_value = mock_response

        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        # Document with duplicate id and lower score should be dropped
        docs = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique"),
        ]

        result = await ranker.run_async(query="test query", documents=docs)

        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["keep me", "unique"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": f"Bearer {ranker.token.resolve_value()}"} if ranker.token.resolve_value() else {},
            attempts=3,
            status_codes_to_retry=None,
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "unique"
        assert result["documents"][1].content == "keep me"

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.huggingface_api.ranker.async_request_with_retry")
    async def test_run_async_empty_documents(self, mock_request, del_hf_env_vars_if_empty):
        """Test run_async with empty documents list"""
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        result = await ranker.run_async(query="test query", documents=[])

        # Check that no API call was made
        mock_request.assert_not_called()
        assert result == {"documents": []}

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("url", "use_grpc"),
        [("http://localhost:8083", False), ("localhost:8084", True)],
    )
    def test_live_run_tei(self, url, use_grpc):
        ranker = HuggingFaceTEIRanker(
            url=url,
            use_grpc=use_grpc,
            token=Secret.from_token("tei-test-key"),
            top_k=1,
        )
        documents = [
            Document(content="Bananas are yellow."),
            Document(content="The capital of France is Paris."),
        ]
        try:
            result = ranker.run(query="What is the capital of France?", documents=documents)
        finally:
            ranker.close()

        assert len(result["documents"]) == 1
        assert result["documents"][0].id == documents[1].id
        assert isinstance(result["documents"][0].score, float)
        assert all(doc.score is None for doc in documents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url", "use_grpc"),
        [("http://localhost:8083", False), ("localhost:8084", True)],
    )
    async def test_live_run_async_tei(self, url, use_grpc):
        ranker = HuggingFaceTEIRanker(
            url=url,
            use_grpc=use_grpc,
            token=Secret.from_token("tei-test-key"),
            top_k=1,
        )
        documents = [
            Document(content="Bananas are yellow."),
            Document(content="The capital of France is Paris."),
        ]
        try:
            result = await ranker.run_async(query="What is the capital of France?", documents=documents)
        finally:
            await ranker.close_async()

        assert len(result["documents"]) == 1
        assert result["documents"][0].id == documents[1].id
        assert isinstance(result["documents"][0].score, float)
        assert all(doc.score is None for doc in documents)
