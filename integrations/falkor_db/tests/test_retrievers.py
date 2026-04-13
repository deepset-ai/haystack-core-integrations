# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack.core.errors import DeserializationError
from haystack.dataclasses import Document
from haystack.document_stores.types.filter_policy import FilterPolicy

from haystack_integrations.components.retrievers.falkor_db import (
    FalkorDBCypherRetriever,
    FalkorDBEmbeddingRetriever,
)
from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore


class TestFalkorDBEmbeddingRetriever:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="must be an instance of FalkorDBDocumentStore"):
            FalkorDBEmbeddingRetriever(document_store=MagicMock())  # type: ignore

    def test_run(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        expected_docs = [Document(content="doc1"), Document(content="doc2")]
        store._embedding_retrieval.return_value = expected_docs

        retriever = FalkorDBEmbeddingRetriever(document_store=store)
        res = retriever.run(query_embedding=[0.1, 0.2])

        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2],
            top_k=10,
            filters=None,
        )
        assert res["documents"] == expected_docs

    def test_filter_policy_replace(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        retriever = FalkorDBEmbeddingRetriever(
            document_store=store,
            filters={"field": "year", "operator": "==", "value": 2020},
            filter_policy=FilterPolicy.REPLACE,
        )

        runtime_filters = {"field": "author", "operator": "==", "value": "Alice"}
        retriever.run(query_embedding=[0.1], filters=runtime_filters)

        # REPLACE policy means runtime filters completely replace init filters
        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1],
            top_k=10,
            filters=runtime_filters,
        )

    def test_filter_policy_merge(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        retriever = FalkorDBEmbeddingRetriever(
            document_store=store,
            filters={"field": "year", "operator": "==", "value": 2020},
            filter_policy=FilterPolicy.MERGE,
        )

        runtime_filters = {"field": "author", "operator": "==", "value": "Alice"}
        retriever.run(query_embedding=[0.1], filters=runtime_filters)

        called_filters = store._embedding_retrieval.call_args[1]["filters"]
        # MERGE policy nests them in an AND
        assert called_filters["operator"] == "AND"
        assert len(called_filters["conditions"]) == 2

    def test_to_dict_from_dict(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        store.to_dict.return_value = {"type": "FalkorDBDocumentStore", "init_parameters": {}}

        retriever = FalkorDBEmbeddingRetriever(
            document_store=store,
            filters={"field": "year", "operator": "==", "value": 2020},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )
        data = retriever.to_dict()
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filter_policy"] == "merge"

        # We can't properly instantiate the mock in from_dict, so we just verify to_dict structure here
        # or we use a real store for the roundtrip:
        store_real = FalkorDBDocumentStore(verify_connectivity=False)
        retriever_real = FalkorDBEmbeddingRetriever(
            document_store=store_real,
            filters={"field": "year", "operator": "==", "value": 2020},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )
        data_real = retriever_real.to_dict()
        new_retriever = FalkorDBEmbeddingRetriever.from_dict(data_real)
        assert new_retriever._top_k == 5
        assert new_retriever._filter_policy == FilterPolicy.MERGE

    def test_from_dict_without_document_store(self):
        data = {"type": "FalkorDBEmbeddingRetriever", "init_parameters": {}}
        with pytest.raises(DeserializationError):
            FalkorDBEmbeddingRetriever.from_dict(data)


class TestFalkorDBCypherRetriever:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="must be an instance of FalkorDBDocumentStore"):
            FalkorDBCypherRetriever(document_store=MagicMock())  # type: ignore

    def test_run_with_init_query(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        expected_docs = [Document(content="doc1")]
        store._cypher_retrieval.return_value = expected_docs

        retriever = FalkorDBCypherRetriever(document_store=store, custom_cypher_query="MATCH (d:Doc) RETURN d")
        res = retriever.run(parameters={"a": 1})

        store._cypher_retrieval.assert_called_once_with(
            cypher_query="MATCH (d:Doc) RETURN d",
            parameters={"a": 1},
        )
        assert res["documents"] == expected_docs

    def test_run_with_runtime_query(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        store._cypher_retrieval.return_value = []

        retriever = FalkorDBCypherRetriever(document_store=store, custom_cypher_query="MATCH (d:Doc) RETURN d")
        # Runtime query overrides init query
        retriever.run(query="MATCH (d:Other) RETURN d")

        store._cypher_retrieval.assert_called_once_with(
            cypher_query="MATCH (d:Other) RETURN d",
            parameters=None,
        )

    def test_run_no_query_raises(self):
        store = MagicMock(spec=FalkorDBDocumentStore)
        retriever = FalkorDBCypherRetriever(document_store=store)

        with pytest.raises(ValueError, match="query string must be provided"):
            retriever.run()

    def test_to_dict_from_dict(self):
        store_real = FalkorDBDocumentStore(verify_connectivity=False)
        retriever = FalkorDBCypherRetriever(
            document_store=store_real,
            custom_cypher_query="MATCH (d) RETURN d",
        )
        data = retriever.to_dict()
        assert data["init_parameters"]["custom_cypher_query"] == "MATCH (d) RETURN d"

        new_retriever = FalkorDBCypherRetriever.from_dict(data)
        assert new_retriever._custom_cypher_query == "MATCH (d) RETURN d"

    def test_from_dict_without_document_store(self):
        data = {"type": "FalkorDBCypherRetriever", "init_parameters": {}}
        with pytest.raises(DeserializationError):
            FalkorDBCypherRetriever.from_dict(data)
