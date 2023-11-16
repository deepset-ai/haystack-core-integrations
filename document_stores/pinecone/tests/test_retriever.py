import os
from inspect import getmembers, isclass, isfunction
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pinecone
import pytest
from haystack.preview import (
    DeserializationError,
    Document,
    component,
    default_from_dict,
    default_to_dict,
)
from haystack.preview.dataclasses import Document

from pinecone_haystack.document_store import PineconeDocumentStore
from pinecone_haystack.retriever import PineconeRetriever
from tests import pinecone_mock


class TestPineconeRetriever:
    @pytest.mark.unit
    def test_init(self):
        mock_store = Mock(spec=PineconeDocumentStore)
        retriever = PineconeRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.filters == None
        assert retriever.top_k == 10
        assert retriever.scale_score == True
        assert retriever.return_embedding == False

    @pytest.mark.unit
    def test_run(self):
        mock_store = Mock(spec=PineconeDocumentStore)
        mock_store.query_by_embedding.return_value = [
            Document(
                content="$TSLA lots of green on the 5 min, watch the hourly $259.33 possible resistance currently @ $257.00.Tesla is recalling 2,700 Model X cars.Hard to find new buyers of $TSLA at 250. Shorts continue to pile in.",
                meta={
                    "target": "TSLA",
                    "sentiment_score": 0.318,
                    "format": "post",
                },
            )
        ]

        retriever = PineconeRetriever(document_store=mock_store)
        results = retriever.run(["How many cars is TSLA recalling?"])

        assert len(results["documents"]) == 1
        assert (
            results["documents"][0].content
            == "$TSLA lots of green on the 5 min, watch the hourly $259.33 possible resistance currently @ $257.00.Tesla is recalling 2,700 Model X cars.Hard to find new buyers of $TSLA at 250. Shorts continue to pile in."
        )

    @pytest.mark.integration
    def test_to_dict(self):
        document_store = PineconeDocumentStore("pinecone-test-key")
        retriever = PineconeRetriever(document_store=document_store)
        doc_dict = retriever.to_dict()
        assert doc_dict == {
            "init_parameters": {
                "document_store": "test_document_store",
                "filters": None,
                "top_k": 10,
                "scale_score": "True",
                "return_embedding": False,
            }
        }

    @pytest.mark.integration
    def test_from_dict(self):
        """
        Test deserialization of this component from a dictionary, using default initialization parameters.
        """
        retriever_component_dict = {
            "type": "PineconeRetriever",
            "init_parameters": {
                "document_store": "test_document_store",
                "filters": None,
                "top_k": 10,
                "scale_score": True,
                "return_embedding": False,
            },
        }
        retriever = PineconeRetriever.from_dict(retriever_component_dict)

        assert retriever.document_store == "test_document_store"
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.scale_score is True
        assert retriever.return_embedding is False
