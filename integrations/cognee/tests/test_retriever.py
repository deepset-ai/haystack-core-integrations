# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from haystack_integrations.components.connectors.cognee import CogneeRetriever


class TestCogneeRetriever:
    def test_init_defaults(self):
        retriever = CogneeRetriever()
        assert retriever.search_type == "GRAPH_COMPLETION"
        assert retriever.top_k == 10
        assert retriever.dataset_name is None

    def test_init_custom(self):
        retriever = CogneeRetriever(search_type="CHUNKS", top_k=5, dataset_name="my_data")
        assert retriever.search_type == "CHUNKS"
        assert retriever.top_k == 5
        assert retriever.dataset_name == "my_data"

    def test_init_invalid_search_type(self):
        with pytest.raises(ValueError, match="Invalid search_type"):
            CogneeRetriever(search_type="INVALID_TYPE")

    def test_to_dict(self):
        retriever = CogneeRetriever(search_type="SUMMARIES", top_k=3, dataset_name="ds")
        data = retriever.to_dict()
        assert data["type"] == "haystack_integrations.components.connectors.cognee.retriever.CogneeRetriever"
        assert data["init_parameters"]["search_type"] == "SUMMARIES"
        assert data["init_parameters"]["top_k"] == 3
        assert data["init_parameters"]["dataset_name"] == "ds"

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.cognee.retriever.CogneeRetriever",
            "init_parameters": {"search_type": "CHUNKS", "top_k": 7, "dataset_name": None},
        }
        retriever = CogneeRetriever.from_dict(data)
        assert retriever.search_type == "CHUNKS"
        assert retriever.top_k == 7

    @patch("haystack_integrations.components.connectors.cognee.retriever.cognee")
    def test_run_returns_documents(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=["result one", "result two"])

        retriever = CogneeRetriever(search_type="GRAPH_COMPLETION", top_k=5)
        result = retriever.run(query="What is Cognee?")

        docs = result["documents"]
        assert len(docs) == 2
        assert docs[0].content == "result one"
        assert docs[0].meta["source"] == "cognee"

    @patch("haystack_integrations.components.connectors.cognee.retriever.cognee")
    def test_run_empty_results(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=[])

        retriever = CogneeRetriever()
        result = retriever.run(query="nonexistent query")

        assert result["documents"] == []

    @patch("haystack_integrations.components.connectors.cognee.retriever.cognee")
    def test_run_respects_top_k_override(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=["a", "b", "c", "d", "e"])

        retriever = CogneeRetriever(top_k=10)
        result = retriever.run(query="test", top_k=2)

        assert len(result["documents"]) == 2

    @patch("haystack_integrations.components.connectors.cognee.retriever.cognee")
    def test_run_handles_dict_results(self, mock_cognee):
        mock_cognee.search = AsyncMock(
            return_value=[
                {"content": "Dict content", "score": 0.9},
                {"text": "Alt text field"},
            ]
        )

        retriever = CogneeRetriever()
        result = retriever.run(query="test")

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Dict content"
        assert result["documents"][1].content == "Alt text field"

    @patch("haystack_integrations.components.connectors.cognee.retriever.cognee")
    def test_run_handles_none_results(self, mock_cognee):
        mock_cognee.search = AsyncMock(return_value=None)

        retriever = CogneeRetriever()
        result = retriever.run(query="test")
        assert result["documents"] == []
