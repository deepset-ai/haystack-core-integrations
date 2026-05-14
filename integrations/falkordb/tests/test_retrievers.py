# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types.filter_policy import FilterPolicy
from redis.exceptions import ResponseError

from haystack_integrations.components.retrievers.falkordb import (
    FalkorDBCypherRetriever,
    FalkorDBEmbeddingRetriever,
)
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore

logger = logging.getLogger(__name__)


class TestFalkorDBEmbeddingRetriever:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="must be an instance of FalkorDBDocumentStore"):
            FalkorDBEmbeddingRetriever(document_store=MagicMock())

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

        data = retriever.to_dict()
        assert data["init_parameters"]["filters"] == {"field": "year", "operator": "==", "value": 2020}
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filter_policy"] == "merge"
        assert isinstance(data["init_parameters"]["document_store"], dict)

        restored = FalkorDBEmbeddingRetriever.from_dict(data)
        assert restored.filters == {"field": "year", "operator": "==", "value": 2020}
        assert restored.top_k == 5
        assert restored.filter_policy == FilterPolicy.MERGE
        assert isinstance(restored.document_store, FalkorDBDocumentStore)

    def test_from_dict_without_document_store(self):
        fqcn = "haystack_integrations.components.retrievers.falkordb"
        fqcn += ".embedding_retriever.FalkorDBEmbeddingRetriever"
        data = {"type": fqcn, "init_parameters": {}}
        with pytest.raises(KeyError):
            FalkorDBEmbeddingRetriever.from_dict(data)


class TestFalkorDBCypherRetriever:
    def test_init_invalid_store(self):
        with pytest.raises(ValueError, match="must be an instance of FalkorDBDocumentStore"):
            FalkorDBCypherRetriever(document_store=MagicMock())

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

        data = retriever.to_dict()
        assert data["init_parameters"]["custom_cypher_query"] == "MATCH (d) RETURN d"
        assert isinstance(data["init_parameters"]["document_store"], dict)

        restored = FalkorDBCypherRetriever.from_dict(data)
        assert restored.custom_cypher_query == "MATCH (d) RETURN d"
        assert isinstance(restored.document_store, FalkorDBDocumentStore)

    def test_from_dict_without_document_store(self):
        data = {
            "type": "haystack_integrations.components.retrievers.falkordb.cypher_retriever.FalkorDBCypherRetriever",
            "init_parameters": {},
        }
        with pytest.raises(KeyError):
            FalkorDBCypherRetriever.from_dict(data)


# ---------------------------------------------------------------------------
# Integration tests — require a live FalkorDB instance.
# Set FALKORDB_HOST / FALKORDB_PORT to point at your server (default: localhost:6379).
# Run with: hatch run test:integration
# ---------------------------------------------------------------------------


def _make_store(graph_name: str, embedding_dim: int = 3) -> FalkorDBDocumentStore:
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    return FalkorDBDocumentStore(
        host=host,
        port=port,
        graph_name=graph_name,
        embedding_dim=embedding_dim,
        recreate_graph=True,
        verify_connectivity=True,
    )


@pytest.mark.integration
class TestFalkorDBEmbeddingRetrieverIntegration:
    """
    Integration tests for FalkorDBEmbeddingRetriever.

    Covers: relevance & correctness, k/pagination, no-results/empty-store,
    result structure, score sanity, metadata filtering, determinism,
    and serialisation round-trip.
    """

    @pytest.fixture
    def document_store(self, request):
        graph_name = f"test_emb_ret_{request.node.name[:40]}"
        store = _make_store(graph_name)
        yield store
        try:
            store.client.select_graph(graph_name).delete()
        except Exception:
            logger.debug("Could not delete graph %s during teardown", graph_name)

    # ------------------------------------------------------------------
    # Relevance & correctness
    # ------------------------------------------------------------------

    def test_run_returns_ranked_documents(self, document_store):
        """Closest document to the query embedding must rank first; all scores in [0, 1]."""
        docs = [
            Document(content="Graph databases represent data as nodes and edges.", embedding=[0.1, 0.2, 0.3]),
            Document(content="Large language models generate text from prompts.", embedding=[0.9, 0.8, 0.1]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=2)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        assert len(result["documents"]) == 2
        assert "Graph databases" in result["documents"][0].content
        for doc in result["documents"]:
            assert doc.score is not None
            assert 0.0 <= doc.score <= 1.0

    # ------------------------------------------------------------------
    # k / Pagination behaviour
    # ------------------------------------------------------------------

    def test_k_equals_one_returns_single_result(self, document_store):
        """top_k=1 must return exactly one document."""
        docs = [
            Document(content="Alpha", embedding=[0.1, 0.1, 0.1]),
            Document(content="Beta", embedding=[0.5, 0.5, 0.5]),
            Document(content="Gamma", embedding=[0.9, 0.9, 0.9]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query_embedding=[0.1, 0.1, 0.1])

        assert len(result["documents"]) == 1

    def test_run_top_k_limits_results(self, document_store):
        """top_k passed at run-time must cap results below the total number of docs."""
        docs = [
            Document(content="Alpha", embedding=[0.1, 0.1, 0.1]),
            Document(content="Beta", embedding=[0.2, 0.2, 0.2]),
            Document(content="Gamma", embedding=[0.3, 0.3, 0.3]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=10)
        result = retriever.run(query_embedding=[0.1, 0.1, 0.1], top_k=2)

        assert len(result["documents"]) <= 2

    def test_k_larger_than_corpus_returns_all_docs(self, document_store):
        """top_k greater than the corpus size must return all documents without crashing."""
        docs = [
            Document(content="Alpha", embedding=[0.1, 0.1, 0.1]),
            Document(content="Beta", embedding=[0.5, 0.5, 0.5]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=100)
        result = retriever.run(query_embedding=[0.1, 0.1, 0.1])

        assert len(result["documents"]) == 2

    def test_k_zero_surfaces_error(self, document_store):
        """top_k=0 is invalid and must surface an error, not silently return results."""
        docs = [Document(content="Any doc.", embedding=[0.1, 0.2, 0.3])]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=1)
        with pytest.raises(ResponseError):
            retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=0)

    # ------------------------------------------------------------------
    # No-results / empty-store edge cases
    # ------------------------------------------------------------------

    def test_empty_store_returns_empty_list(self, document_store):
        """Querying an empty store must return an empty list, not raise."""
        retriever = FalkorDBEmbeddingRetriever(document_store=document_store)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        assert result["documents"] == []

    def test_wrong_embedding_dimension_surfaces_error(self, document_store):
        """A query vector whose dimension mismatches embedding_dim must surface an error cleanly."""
        docs = [Document(content="Valid doc.", embedding=[0.1, 0.2, 0.3])]
        document_store.write_documents(docs)

        # Store was built with embedding_dim=3; passing 768 floats is a hard mismatch.
        retriever = FalkorDBEmbeddingRetriever(document_store=document_store)
        with pytest.raises(ResponseError):
            retriever.run(query_embedding=[0.1] * 768)

    # ------------------------------------------------------------------
    # Result structure
    # ------------------------------------------------------------------

    def test_result_has_required_fields(self, document_store):
        """Every retrieved document must carry id, content, meta, and a numeric score."""
        docs = [Document(content="Structured result doc.", embedding=[0.3, 0.3, 0.3], meta={"source": "test"})]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store)
        result = retriever.run(query_embedding=[0.3, 0.3, 0.3])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.id is not None
        assert doc.content == "Structured result doc."
        assert isinstance(doc.score, float)
        assert doc.meta.get("source") == "test"

    def test_scores_sorted_descending(self, document_store):
        """Results must be ordered from most similar to least similar (score descending)."""
        docs = [
            Document(content="Very close.", embedding=[0.1, 0.2, 0.3]),
            Document(content="Somewhat related.", embedding=[0.4, 0.5, 0.6]),
            Document(content="Completely different.", embedding=[0.9, 0.0, 0.0]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=3)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        scores = [doc.score for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"

    def test_no_duplicate_documents_in_results(self, document_store):
        """The result set must not contain duplicate document IDs."""
        docs = [
            Document(content="Unique doc one.", embedding=[0.1, 0.1, 0.1]),
            Document(content="Unique doc two.", embedding=[0.2, 0.2, 0.2]),
            Document(content="Unique doc three.", embedding=[0.3, 0.3, 0.3]),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=10)
        result = retriever.run(query_embedding=[0.1, 0.1, 0.1])

        ids = [doc.id for doc in result["documents"]]
        assert len(ids) == len(set(ids)), f"Duplicate document IDs found: {ids}"

    # ------------------------------------------------------------------
    # Score sanity
    # ------------------------------------------------------------------

    def test_identical_query_scores_near_maximum(self, document_store):
        """Querying with the same vector as a stored document must yield a score near 1.0."""
        embedding = [0.1, 0.2, 0.3]
        docs = [Document(content="Self-retrieval target.", embedding=embedding)]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query_embedding=embedding)

        assert len(result["documents"]) == 1
        # vecf32 precision loss may cause a tiny deviation from 1.0; allow >= 0.99
        assert result["documents"][0].score >= 0.99

    def test_antiparallel_query_scores_near_minimum(self, document_store):
        """An anti-parallel query vector must yield a score near 0.0 (maximum cosine distance).

        FalkorDB returns cosine distance in [0, 2]; the scaling formula is 1 - (d / 2).
        Anti-parallel unit vectors have cosine similarity = -1, distance = 2, scaled score = 0.0.
        """
        docs = [Document(content="Anti-parallel target.", embedding=[1.0, 0.0, 0.0])]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store, top_k=1)
        result = retriever.run(query_embedding=[-1.0, 0.0, 0.0])

        assert len(result["documents"]) == 1
        # Allow a small tolerance for vecf32 precision loss
        assert result["documents"][0].score <= 0.05

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    def test_run_with_metadata_filter(self, document_store):
        """Metadata filter must narrow the result set to matching documents only."""
        docs = [
            Document(content="Graph doc.", embedding=[0.1, 0.2, 0.3], meta={"category": "graph"}),
            Document(content="ML doc.", embedding=[0.1, 0.2, 0.3], meta={"category": "ml"}),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBEmbeddingRetriever(document_store=document_store)
        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"field": "category", "operator": "==", "value": "graph"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["category"] == "graph"

    # ------------------------------------------------------------------
    # Serialisation round-trip
    # ------------------------------------------------------------------

    def test_to_dict_from_dict_roundtrip(self, document_store):
        """Serialisation round-trip must preserve all retriever configuration."""
        retriever = FalkorDBEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "year", "operator": "==", "value": 2024},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )

        restored = FalkorDBEmbeddingRetriever.from_dict(retriever.to_dict())

        assert restored.top_k == 5
        assert restored.filter_policy == FilterPolicy.MERGE
        assert restored.filters == {"field": "year", "operator": "==", "value": 2024}


@pytest.mark.integration
class TestFalkorDBCypherRetrieverIntegration:
    """
    Integration tests for FalkorDBCypherRetriever.

    Covers: correctness, graph traversal, no-results/empty-store, edge-case
    inputs (empty string, special characters), result structure, determinism,
    runtime query override, and serialisation round-trip.
    """

    @pytest.fixture
    def document_store(self, request):
        graph_name = f"test_cypher_ret_{request.node.name[:40]}"
        store = _make_store(graph_name)
        yield store
        try:
            store.client.select_graph(graph_name).delete()
        except Exception:
            logger.debug("Could not delete graph %s during teardown", graph_name)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_run_match_all_documents(self, document_store):
        """A MATCH-all query must return every document written to the store."""
        docs = [
            Document(content="First document"),
            Document(content="Second document"),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d ORDER BY d.id",
        )
        result = retriever.run()

        assert len(result["documents"]) == 2
        contents = {doc.content for doc in result["documents"]}
        assert contents == {"First document", "Second document"}

    def test_run_with_parameters(self, document_store):
        """A parametrised query must filter results to the matching document only."""
        docs = [
            Document(content="Alpha doc", meta={"tag": "alpha"}),
            Document(content="Beta doc", meta={"tag": "beta"}),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document {tag: $tag}) RETURN d",
        )
        result = retriever.run(parameters={"tag": "alpha"})

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["tag"] == "alpha"

    def test_run_graph_traversal(self, document_store):
        """A relationship-traversal query must follow edges and return the target node."""
        document_store._ensure_connected()
        document_store.graph.query(
            "CREATE (a:Document {id: 'src', content: 'Source Doc'})"
            "-[:CITES]->"
            "(b:Document {id: 'cited', content: 'Cited Doc'})"
        )

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (:Document {id: $source_id})-[:CITES]->(t:Document) RETURN t",
        )
        result = retriever.run(parameters={"source_id": "src"})

        assert len(result["documents"]) == 1
        assert result["documents"][0].id == "cited"
        assert result["documents"][0].content == "Cited Doc"

    # ------------------------------------------------------------------
    # No-results / empty-store edge cases
    # ------------------------------------------------------------------

    def test_empty_store_returns_empty_list(self, document_store):
        """Querying an empty store must return an empty list, not raise."""
        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )
        result = retriever.run()

        assert result["documents"] == []

    def test_query_matching_nothing_returns_empty_list(self, document_store):
        """A valid query that matches no nodes must return an empty list, not raise."""
        docs = [Document(content="Present doc", meta={"status": "active"})]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document {status: $status}) RETURN d",
        )
        result = retriever.run(parameters={"status": "archived"})

        assert result["documents"] == []

    # ------------------------------------------------------------------
    # Edge-case inputs
    # ------------------------------------------------------------------

    def test_empty_string_query_falls_back_to_init_query(self, document_store):
        """An empty-string run-time query is falsy and must fall back to custom_cypher_query."""
        docs = [Document(content="Fallback target")]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )
        result = retriever.run(query="")

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Fallback target"

    def test_special_characters_in_parameters(self, document_store):
        """Parameters with unicode and punctuation must be stored and matched correctly."""
        special = "日本語 & <spéciäl> chars: 100% safe"
        docs = [Document(content="Unicode doc", meta={"label": special})]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document {label: $label}) RETURN d",
        )
        result = retriever.run(parameters={"label": special})

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["label"] == special

    # ------------------------------------------------------------------
    # Result structure
    # ------------------------------------------------------------------

    def test_result_has_required_fields(self, document_store):
        """Every retrieved document must carry id, content, and meta."""
        docs = [Document(content="Structured doc.", meta={"key": "value"})]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )
        result = retriever.run()

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.id is not None
        assert doc.content == "Structured doc."
        assert doc.meta.get("key") == "value"

    def test_no_duplicate_documents_in_results(self, document_store):
        """Results from a MATCH query must not contain duplicate document IDs."""
        docs = [
            Document(content="Unique one"),
            Document(content="Unique two"),
            Document(content="Unique three"),
        ]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )
        result = retriever.run()

        ids = [doc.id for doc in result["documents"]]
        assert len(ids) == len(set(ids)), f"Duplicate document IDs found: {ids}"

    # ------------------------------------------------------------------
    # Runtime query override
    # ------------------------------------------------------------------

    def test_run_runtime_query_override(self, document_store):
        """A query passed at run-time must override the custom_cypher_query set at init."""
        docs = [Document(content="Only doc")]
        document_store.write_documents(docs)

        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )
        result = retriever.run(query="MATCH (d:Other) RETURN d")

        assert result["documents"] == []

    # ------------------------------------------------------------------
    # Serialisation round-trip
    # ------------------------------------------------------------------

    def test_to_dict_from_dict_roundtrip(self, document_store):
        """Serialisation round-trip must preserve the custom_cypher_query."""
        retriever = FalkorDBCypherRetriever(
            document_store=document_store,
            custom_cypher_query="MATCH (d:Document) RETURN d",
        )

        restored = FalkorDBCypherRetriever.from_dict(retriever.to_dict())

        assert restored.custom_cypher_query == "MATCH (d:Document) RETURN d"
