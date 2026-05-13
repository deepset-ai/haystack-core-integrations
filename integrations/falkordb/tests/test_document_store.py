# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import falkordb as _falkordb_module
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.components.retrievers.falkordb import (
    FalkorDBCypherRetriever,
    FalkorDBEmbeddingRetriever,
)
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack_integrations.document_stores.falkordb.document_store import (
    _convert_filters,
)

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


def _result(rows):
    return MagicMock(result_set=rows)


@pytest.fixture
def mock_falkordb(monkeypatch):
    constructor = MagicMock()
    client = MagicMock()
    graph = MagicMock()
    constructor.return_value = client
    client.select_graph.return_value = graph
    graph.query.return_value = _result([])
    monkeypatch.setattr(_falkordb_module, "FalkorDB", constructor)
    return constructor, client, graph


class TestFalkorDBDocumentStoreUnit:
    def test_init_rejects_invalid_similarity(self):
        with pytest.raises(ValueError, match="not supported"):
            FalkorDBDocumentStore(similarity="invalid")

    @pytest.mark.parametrize(
        "filter_node, expected_clause, expected_params",
        [
            ({"field": "year", "operator": "==", "value": 2024}, "coalesce(d.year = $p0, false)", {"p0": 2024}),
            ({"field": "year", "operator": "==", "value": None}, "d.year IS NULL", {}),
            ({"field": "y", "operator": "!=", "value": None}, "d.y IS NOT NULL", {}),
            ({"field": "y", "operator": ">", "value": 1}, "coalesce(d.y > $p0, false)", {"p0": 1}),
            ({"field": "y", "operator": ">", "value": None}, "false", {}),
            (
                {"field": "tag", "operator": "in", "value": ["a"]},
                "coalesce(d.tag IN $p0, false)",
                {"p0": ["a"]},
            ),
            (
                {"field": "tag", "operator": "not in", "value": ["a"]},
                "coalesce(NOT (d.tag IN $p0), true)",
                {"p0": ["a"]},
            ),
            (
                {"field": "meta.year", "operator": "==", "value": 2024},
                "coalesce(d.year = $p0, false)",
                {"p0": 2024},
            ),
            (
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "y", "operator": "==", "value": 1},
                        {"operator": "NOT", "conditions": [{"field": "b", "operator": "<=", "value": 0}]},
                    ],
                },
                "(coalesce(d.y = $p0, false) OR NOT (coalesce(d.b <= $p1, false)))",
                {"p0": 1, "p1": 0},
            ),
        ],
    )
    def test_convert_filters(self, filter_node, expected_clause, expected_params):
        clause, params = _convert_filters(filter_node)
        assert clause == expected_clause
        assert params == expected_params

    @pytest.mark.parametrize(
        "filter_node, match",
        [
            ({"operator": "AND"}, "requires a 'conditions' key"),
            ({"operator": "NOT"}, "requires a 'conditions' key"),
            ({"operator": "==", "value": 1}, "requires a 'field' key"),
            ({"operator": "==", "field": "f"}, "requires a 'value' key"),
            ({"field": "x", "operator": ">", "value": [1, 2]}, "does not support list values"),
            ({"field": "x", "operator": ">", "value": "not-a-date"}, "non-ISO string"),
            ({"field": "x", "operator": "in", "value": "scalar"}, "requires a list value"),
            ({"field": "x", "operator": "not in", "value": "scalar"}, "requires a list value"),
            ({"field": "x", "operator": "regex", "value": "."}, "Unsupported filter operator"),
        ],
    )
    def test_convert_filters_errors(self, filter_node, match):
        with pytest.raises(FilterError, match=match):
            _convert_filters(filter_node)

    @pytest.mark.parametrize("rows, expected", [([[42]], 42), ([], 0)])
    def test_count_documents(self, mock_falkordb, rows, expected):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result(rows)]
        assert FalkorDBDocumentStore().count_documents() == expected

    def test_filter_documents_no_filters(self, mock_falkordb):
        _, _, graph = mock_falkordb
        node = SimpleNamespace(properties={"id": "n1", "content": "hello"})
        graph.query.side_effect = [_result([]), _result([]), _result([[node]])]
        docs = FalkorDBDocumentStore().filter_documents()
        assert [d.content for d in docs] == ["hello"]
        assert "WHERE" not in graph.query.call_args_list[-1].args[0]

    def test_filter_documents_with_filters(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([])]
        FalkorDBDocumentStore().filter_documents({"field": "year", "operator": "==", "value": 2024})
        last = graph.query.call_args_list[-1]
        assert "WHERE" in last.args[0]
        assert last.args[1] == {"p0": 2024}

    @pytest.mark.usefixtures("mock_falkordb")
    def test_filter_documents_malformed_raises(self):
        with pytest.raises(FilterError, match="Invalid filter syntax"):
            FalkorDBDocumentStore().filter_documents({"field": "year", "value": 2024})

    def test_delete_documents_empty_is_noop(self, mock_falkordb):
        _, _, graph = mock_falkordb
        FalkorDBDocumentStore().delete_documents([])
        assert graph.query.call_count == 2

    def test_delete_documents_runs_query(self, mock_falkordb):
        _, _, graph = mock_falkordb
        FalkorDBDocumentStore().delete_documents(["a", "b"])
        last = graph.query.call_args_list[-1]
        assert "DETACH DELETE" in last.args[0]
        assert last.args[1] == {"ids": ["a", "b"]}

    @pytest.mark.parametrize(
        "similarity, raw, scale_score, filters, expected_score, where_expected",
        [
            ("cosine", 0.4, True, None, 0.8, False),
            ("cosine", 0.4, False, None, 0.4, False),
            ("euclidean", 1.0, True, {"field": "y", "operator": "==", "value": 1}, 0.5, True),
        ],
    )
    def test_embedding_retrieval(
        self, mock_falkordb, similarity, raw, scale_score, filters, expected_score, where_expected
    ):
        _, _, graph = mock_falkordb
        node = SimpleNamespace(properties={"id": "n1", "content": "hello"})
        graph.query.side_effect = [_result([]), _result([]), _result([[node, raw]])]
        docs = FalkorDBDocumentStore(similarity=similarity)._embedding_retrieval(
            query_embedding=[0.1], top_k=5, filters=filters, scale_score=scale_score
        )
        assert docs[0].score == pytest.approx(expected_score)
        assert ("WHERE" in graph.query.call_args_list[-1].args[0]) is where_expected

    def test_cypher_retrieval_returns_documents(self, mock_falkordb):
        _, _, graph = mock_falkordb
        node = SimpleNamespace(properties={"id": "n1", "content": "hello"})
        graph.query.side_effect = [_result([]), _result([]), _result([[node]])]
        docs = FalkorDBDocumentStore()._cypher_retrieval("MATCH (d) RETURN d", parameters={"k": 1})
        assert docs[0].content == "hello"

    @pytest.mark.usefixtures("mock_falkordb")
    def test_write_documents_rejects_non_documents(self):
        with pytest.raises(ValueError, match="expects a list of Documents"):
            FalkorDBDocumentStore().write_documents(["not a doc"])

    @pytest.mark.usefixtures("mock_falkordb")
    def test_write_documents_empty_list_returns_zero(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert FalkorDBDocumentStore().write_documents([]) == 0
        assert "empty list" in caplog.text

    def test_write_documents_policy_none_coerced_to_fail(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([["a"]])]
        with pytest.raises(DuplicateDocumentError):
            FalkorDBDocumentStore().write_documents([Document(id="a", content="x")])

    def test_write_documents_drops_duplicates_within_batch(self, mock_falkordb, caplog):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([]), _result([[2]])]
        with caplog.at_level(logging.INFO):
            written = FalkorDBDocumentStore().write_documents(
                [
                    Document(id="a", content="x"),
                    Document(id="a", content="x"),
                    Document(id="b", content="y"),
                ],
                policy=DuplicatePolicy.SKIP,
            )
        assert written == 2
        assert "already present in the batch" in caplog.text
        sent_ids = [d["id"] for d in graph.query.call_args_list[-1].args[1]["docs"]]
        assert sent_ids == ["a", "b"]

    def test_write_documents_skip_filters_existing(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([["a"]]), _result([[1]])]
        written = FalkorDBDocumentStore().write_documents(
            [Document(id="a", content="x"), Document(id="b", content="y")],
            policy=DuplicatePolicy.SKIP,
        )
        assert written == 1
        sent_ids = [d["id"] for d in graph.query.call_args_list[-1].args[1]["docs"]]
        assert sent_ids == ["b"]

    def test_write_documents_overwrite_uses_on_match_set(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([[1]])]
        FalkorDBDocumentStore().write_documents([Document(id="a", content="x")], policy=DuplicatePolicy.OVERWRITE)
        assert "ON MATCH SET d = doc" in graph.query.call_args_list[-1].args[0]

    def test_write_documents_embeddings_second_pass(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), _result([[1]]), _result([])]
        FalkorDBDocumentStore().write_documents(
            [Document(id="a", content="x", embedding=[0.1, 0.2, 0.3])],
            policy=DuplicatePolicy.OVERWRITE,
        )
        last = graph.query.call_args_list[-1]
        assert "vecf32" in last.args[0]
        assert last.args[1]["docs"] == [{"id": "a", "emb": [0.1, 0.2, 0.3]}]

    def test_write_documents_wraps_errors(self, mock_falkordb):
        _, _, graph = mock_falkordb
        graph.query.side_effect = [_result([]), _result([]), Exception("boom")]
        with pytest.raises(DocumentStoreError, match="Failed to write documents"):
            FalkorDBDocumentStore().write_documents([Document(id="a", content="x")], policy=DuplicatePolicy.OVERWRITE)


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
