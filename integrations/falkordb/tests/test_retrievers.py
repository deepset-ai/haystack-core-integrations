# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types.filter_policy import FilterPolicy

from haystack_integrations.components.retrievers.falkordb import (
    FalkorDBCypherRetriever,
    FalkorDBEmbeddingRetriever,
)
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore


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
        store = FalkorDBDocumentStore(verify_connectivity=False)
        retriever = FalkorDBEmbeddingRetriever(
            document_store=store,
            filters={"field": "year", "operator": "==", "value": 2020},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )

        store_data = default_to_dict(
            store,
            host=store.host,
            port=store.port,
            graph_name=store.graph_name,
            username=store.username,
            password=store.password,
            node_label=store.node_label,
            embedding_dim=store.embedding_dim,
            embedding_field=store.embedding_field,
            similarity=store.similarity,
            write_batch_size=store.write_batch_size,
            recreate_graph=store.recreate_graph,
            verify_connectivity=store.verify_connectivity,
        )
        data = default_to_dict(
            retriever,
            document_store=store_data,
            filters={"field": "year", "operator": "==", "value": 2020},
            top_k=5,
            filter_policy=FilterPolicy.MERGE.value,
        )
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filter_policy"] == "merge"

        data["init_parameters"]["document_store"] = default_from_dict(
            FalkorDBDocumentStore, data["init_parameters"]["document_store"]
        )
        restored = default_from_dict(FalkorDBEmbeddingRetriever, data)
        assert restored.top_k == 5
        assert restored.filter_policy == FilterPolicy.MERGE

    def test_from_dict_without_document_store(self):
        fqcn = "haystack_integrations.components.retrievers.falkordb"
        fqcn += ".embedding_retriever.FalkorDBEmbeddingRetriever"
        data = {"type": fqcn, "init_parameters": {}}
        with pytest.raises(TypeError):
            default_from_dict(FalkorDBEmbeddingRetriever, data)


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
        store = FalkorDBDocumentStore(verify_connectivity=False)
        retriever = FalkorDBCypherRetriever(
            document_store=store,
            custom_cypher_query="MATCH (d) RETURN d",
        )

        store_data = default_to_dict(
            store,
            host=store.host,
            port=store.port,
            graph_name=store.graph_name,
            username=store.username,
            password=store.password,
            node_label=store.node_label,
            embedding_dim=store.embedding_dim,
            embedding_field=store.embedding_field,
            similarity=store.similarity,
            write_batch_size=store.write_batch_size,
            recreate_graph=store.recreate_graph,
            verify_connectivity=store.verify_connectivity,
        )
        data = default_to_dict(
            retriever,
            document_store=store_data,
            custom_cypher_query="MATCH (d) RETURN d",
        )
        assert data["init_parameters"]["custom_cypher_query"] == "MATCH (d) RETURN d"

        data["init_parameters"]["document_store"] = default_from_dict(
            FalkorDBDocumentStore, data["init_parameters"]["document_store"]
        )
        restored = default_from_dict(FalkorDBCypherRetriever, data)
        assert restored.custom_cypher_query == "MATCH (d) RETURN d"

    def test_from_dict_without_document_store(self):
        data = {
            "type": "haystack_integrations.components.retrievers.falkordb.cypher_retriever.FalkorDBCypherRetriever",
            "init_parameters": {},
        }
        with pytest.raises(TypeError):
            default_from_dict(FalkorDBCypherRetriever, data)
