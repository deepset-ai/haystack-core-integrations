# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore
from haystack_integrations.components.retrievers.falkor_db import (
    FalkorDBEmbeddingRetriever,
    FalkorDBCypherRetriever,
)


@pytest.fixture
def falkordb_document_store():
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))

    # recreate_index=True drops the graph data on init
    store = FalkorDBDocumentStore(
        host=host, port=port, graph_name="haystack_test", recreate_index=True, verify_connectivity=True
    )
    yield store

    # Teardown
    store._g.query(f"MATCH (n) DETACH DELETE n")


@pytest.mark.integration
class TestFalkorDBIntegration:
    def test_write_and_filter_documents(self, falkordb_document_store: FalkorDBDocumentStore):
        docs = [
            Document(content="Doc A", meta={"category": "news", "year": 2023}),
            Document(content="Doc B", meta={"category": "blog", "year": 2024}),
            Document(content="Doc C", meta={"category": "news", "year": 2024}),
        ]

        written = falkordb_document_store.write_documents(docs)
        assert written == 3
        assert falkordb_document_store.count_documents() == 3

        # Filter docs
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "category", "operator": "==", "value": "news"},
                {"field": "year", "operator": "==", "value": 2024},
            ],
        }
        filtered = falkordb_document_store.filter_documents(filters)
        assert len(filtered) == 1
        assert filtered[0].content == "Doc C"

    def test_write_documents_duplicate_skip(self, falkordb_document_store: FalkorDBDocumentStore):
        doc = Document(id="doc_1", content="First version")
        falkordb_document_store.write_documents([doc])
        assert falkordb_document_store.count_documents() == 1

        doc_dupe = Document(id="doc_1", content="Second version (skipped)")
        written = falkordb_document_store.write_documents([doc_dupe], policy=DuplicatePolicy.SKIP)
        assert written == 0
        assert falkordb_document_store.count_documents() == 1

        # Verify it wasn't overwritten
        res = falkordb_document_store.filter_documents()
        assert res[0].content == "First version"

    def test_write_documents_duplicate_overwrite(self, falkordb_document_store: FalkorDBDocumentStore):
        doc = Document(id="doc_1", content="First version", meta={"tag": "old"})
        falkordb_document_store.write_documents([doc])

        doc_dupe = Document(id="doc_1", content="Second version", meta={"tag": "new"})
        written = falkordb_document_store.write_documents([doc_dupe], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        assert falkordb_document_store.count_documents() == 1

        # Verify it was overwritten
        res = falkordb_document_store.filter_documents()
        assert res[0].content == "Second version"
        assert res[0].meta["tag"] == "new"

    def test_embedding_retrieval(self, falkordb_document_store: FalkorDBDocumentStore):
        docs = [
            Document(content="Graph databases represent data as nodes and edges.", embedding=[0.1, 0.2, 0.3]),
            Document(content="Large language models generate text.", embedding=[0.9, 0.8, 0.1]),
        ]
        falkordb_document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=falkordb_document_store, top_k=1)
        res = retriever.run(query_embedding=[0.1, 0.25, 0.3])

        retrieved_docs = res["documents"]
        assert len(retrieved_docs) == 1
        assert "Graph databases" in retrieved_docs[0].content
        assert retrieved_docs[0].score is not None

    def test_cypher_retriever_graph_traversal(self, falkordb_document_store: FalkorDBDocumentStore):
        # We manually seed the graph with a relationship
        falkordb_document_store._g.query(
            "CREATE (a:Document {id: 'docA', content: 'Node A'})"
            "-[:REFERENCES]->"
            "(b:Document {id: 'docB', content: 'Node B'})"
        )

        # We use cypher retriever to fetch the referenced document
        retriever = FalkorDBCypherRetriever(
            document_store=falkordb_document_store,
            custom_cypher_query="MATCH (:Document {id: $source_id})-[:REFERENCES]->(target:Document) RETURN target",
        )
        res = retriever.run(parameters={"source_id": "docA"})

        docs = res["documents"]
        assert len(docs) == 1
        assert docs[0].id == "docB"
        assert docs[0].content == "Node B"

    def test_delete_documents(self, falkordb_document_store: FalkorDBDocumentStore):
        docs = [
            Document(id="doc_1", content="One"),
            Document(id="doc_2", content="Two"),
        ]
        falkordb_document_store.write_documents(docs)
        assert falkordb_document_store.count_documents() == 2

        falkordb_document_store.delete_documents(["doc_1"])
        assert falkordb_document_store.count_documents() == 1

        res = falkordb_document_store.filter_documents()
        assert res[0].id == "doc_2"
