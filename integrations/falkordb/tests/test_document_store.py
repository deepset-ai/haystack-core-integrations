# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest
from haystack.dataclasses import Document
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.components.retrievers.falkordb import (
    FalkorDBCypherRetriever,
    FalkorDBEmbeddingRetriever,
)
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore

logger = logging.getLogger(__name__)


class TestFalkorDBDocumentStoreSerialization:
    def test_to_dict_from_dict(self):
        store = FalkorDBDocumentStore(
            host="myhost",
            port=1234,
            graph_name="test_graph",
            embedding_dim=512,
            similarity="euclidean",
            verify_connectivity=False,
        )

        data = store.to_dict()

        assert data["init_parameters"]["host"] == "myhost"
        assert data["init_parameters"]["port"] == 1234
        assert data["init_parameters"]["embedding_dim"] == 512
        assert data["init_parameters"]["similarity"] == "euclidean"

        restored = FalkorDBDocumentStore.from_dict(data)
        assert restored.host == "myhost"
        assert restored.port == 1234
        assert restored.graph_name == "test_graph"
        assert restored.embedding_dim == 512
        assert restored.similarity == "euclidean"


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Test FalkorDBDocumentStore against the standard Haystack DocumentStore tests.
    """

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        FalkorDB stores embeddings as vecf32 (float32), so exact float64 round-trip
        equality is not possible. Sort both lists by id to compensate for non-deterministic
        graph traversal order, and compare only id/content/meta plus embedding presence.
        """
        assert len(received) == len(expected), f"Expected {len(expected)} documents but got {len(received)}"
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)
        for recv, exp in zip(received_sorted, expected_sorted, strict=True):
            assert recv.id == exp.id
            assert recv.content == exp.content
            assert recv.meta == exp.meta
            assert (recv.embedding is None) == (exp.embedding is None)

    @pytest.fixture
    def document_store(self, request):
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))

        # Use a unique graph name for each test to avoid interference
        graph_name = f"test_graph_{request.node.name[:30]}"
        store = FalkorDBDocumentStore(
            host=host,
            port=port,
            graph_name=graph_name,
            embedding_dim=768,
            recreate_graph=True,
            verify_connectivity=True,
        )
        yield store
        # Teardown: delete the graph
        try:
            store.client.select_graph(graph_name).delete()
        except Exception:
            logger.debug("Could not delete graph %s during teardown", graph_name)

    def test_write_documents(self, document_store):
        """
        Test write_documents() default behaviour.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    @pytest.fixture
    def embedding_store(self):
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))
        store = FalkorDBDocumentStore(
            host=host,
            port=port,
            graph_name="test_embedding_retrieval",
            embedding_dim=3,
            recreate_graph=True,
            verify_connectivity=True,
        )
        yield store
        try:
            store.client.select_graph("test_embedding_retrieval").delete()
        except Exception:
            logger.debug("Could not delete graph test_embedding_retrieval during teardown")

    def test_embedding_retrieval(self, embedding_store):
        docs = [
            Document(content="Graph databases represent data as nodes and edges.", embedding=[0.1, 0.2, 0.3]),
            Document(content="Large language models generate text.", embedding=[0.9, 0.8, 0.1]),
        ]
        embedding_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=embedding_store, top_k=1)
        res = retriever.run(query_embedding=[0.1, 0.25, 0.3])

        assert len(res["documents"]) == 1
        assert "Graph databases" in res["documents"][0].content
        assert res["documents"][0].score is not None

    def test_cypher_retriever_graph_traversal(self, document_store):
        document_store.graph.query(
            "CREATE (a:Document {id: 'docA', content: 'Node A'})"
            "-[:REFERENCES]->"
            "(b:Document {id: 'docB', content: 'Node B'})"
        )

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query=("MATCH (:Document {id: $source_id})-[:REFERENCES]->(target:Document) RETURN target"),
        )
        res = retriever.run(parameters={"source_id": "docA"})

        assert len(res["documents"]) == 1
        assert res["documents"][0].id == "docB"
        assert res["documents"][0].content == "Node B"
