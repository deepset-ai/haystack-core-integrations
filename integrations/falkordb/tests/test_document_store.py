# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore


@pytest.fixture
def mock_falkordb():
    with patch("falkordb.FalkorDB") as mock_client_class:
        mock_client = MagicMock()
        mock_graph = MagicMock()
        mock_client.select_graph.return_value = mock_graph
        mock_client_class.return_value = mock_client
        yield mock_client_class, mock_client, mock_graph


class TestFalkorDBDocumentStore:
    def test_init_invalid_similarity(self):
        with pytest.raises(ValueError, match="is not supported by FalkorDBDocumentStore"):
            FalkorDBDocumentStore(similarity="dot_product")  # type: ignore

    @pytest.mark.usefixtures("mock_falkordb")
    def test_write_documents_empty(self):
        store = FalkorDBDocumentStore()
        assert store.write_documents([]) == 0

    @pytest.mark.usefixtures("mock_falkordb")
    def test_write_documents_invalid_type(self):
        store = FalkorDBDocumentStore()
        with pytest.raises(ValueError, match="expects a list of Documents"):
            store.write_documents([{"content": "not a document"}])  # type: ignore

    def test_write_documents_overwrite_policy(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        doc = Document(id="1", content="Hello", meta={"year": 2024})

        def mock_query(cypher, _params=None):
            m = MagicMock()
            if "RETURN d.id" in cypher:
                m.result_set = []
            elif "MERGE" in cypher:
                m.result_set = [[1]]
            else:
                m.result_set = []
            return m

        mock_graph.query.side_effect = mock_query

        written = store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1

        # Find the MERGE query
        write_call = next(call for call in mock_graph.query.call_args_list if "MERGE" in call[0][0])
        assert "ON MATCH SET d += doc" in write_call[0][0]

    def test_write_documents_skip_policy(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        doc = Document(id="1", content="Hello")

        def mock_query(cypher, _params=None):
            m = MagicMock()
            if "RETURN d.id" in cypher:
                m.result_set = [["1"]]  # ID already exists
            else:
                m.result_set = []
            return m

        mock_graph.query.side_effect = mock_query

        written = store.write_documents([doc], policy=DuplicatePolicy.SKIP)
        assert written == 0

    def test_write_documents_fail_policy(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        doc = Document(id="1", content="Hello")

        def mock_query(cypher, _params=None):
            m = MagicMock()
            if "RETURN d.id" in cypher:
                m.result_set = [["1"]]
            else:
                m.result_set = []
            return m

        mock_graph.query.side_effect = mock_query

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_write_documents_batching(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore(write_batch_size=2)
        docs = [
            Document(id="1", content="Doc 1"),
            Document(id="2", content="Doc 2"),
            Document(id="3", content="Doc 3"),
        ]

        def mock_query(cypher, params=None):
            m = MagicMock()
            if "RETURN d.id" in cypher:
                m.result_set = []
            elif "MERGE" in cypher:
                m.result_set = [[len(params["docs"])]] if params and "docs" in params else [[1]]
            else:
                m.result_set = []
            return m

        mock_graph.query.side_effect = mock_query

        written = store.write_documents(docs, policy=DuplicatePolicy.FAIL)
        assert written == 3

        write_calls = [call for call in mock_graph.query.call_args_list if "MERGE" in call[0][0]]
        assert len(write_calls) == 2  # 3 docs, size 2 -> 2 batches

    def test_filter_documents_empty_filters(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_res = MagicMock()
        mock_res.result_set = []
        mock_graph.query.return_value = mock_res

        store.filter_documents()

        cypher = mock_graph.query.call_args_list[-1][0][0]
        assert "WHERE" not in cypher
        assert "MATCH (d:Document) RETURN d" in cypher

    @pytest.mark.usefixtures("mock_falkordb")
    def test_filter_documents_invalid(self):
        store = FalkorDBDocumentStore()
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            store.filter_documents({"wrong": "syntax"})

    def test_filter_documents_eq_operator(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_res = MagicMock()
        mock_res.result_set = []
        mock_graph.query.return_value = mock_res

        store.filter_documents({"field": "author", "operator": "==", "value": "John"})

        cypher, params = mock_graph.query.call_args[0]
        assert "WHERE" in cypher
        assert "d.author = " in cypher
        assert params["p0"] == "John"

    def test_filter_documents_and_operator(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_res = MagicMock()
        mock_res.result_set = []
        mock_graph.query.return_value = mock_res

        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "year", "operator": ">=", "value": 2020},
                {"field": "year", "operator": "<", "value": 2025},
            ],
        }
        store.filter_documents(filters)

        cypher, params = mock_graph.query.call_args[0]
        assert "WHERE (d.year >= $p0 AND d.year < $p1)" in cypher
        assert params["p0"] == 2020
        assert params["p1"] == 2025

    def test_delete_documents(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        store.delete_documents(["1", "2"])

        cypher, params = mock_graph.query.call_args[0]
        assert "DETACH DELETE d" in cypher
        assert params["ids"] == ["1", "2"]

    def test_delete_documents_empty(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        store.delete_documents([])
        assert mock_graph.query.call_count == 2

    def test_count_documents(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_res = MagicMock()
        mock_res.result_set = [[42]]
        mock_graph.query.return_value = mock_res

        assert store.count_documents() == 42

        cypher = mock_graph.query.call_args[0][0]
        assert "RETURN count(d) AS n" in cypher

    def test_to_dict_from_dict_roundtrip(self):
        store = FalkorDBDocumentStore(
            host="falkor.test",
            port=9999,
            graph_name="my_graph",
            username="admin",
            password=Secret.from_env_var("FALKOR_PWD", strict=False),
            node_label="CustomDoc",
            embedding_dim=1536,
            embedding_field="vector",
            similarity="euclidean",
            write_batch_size=50,
            recreate_index=True,
            verify_connectivity=False,
        )
        data = store.to_dict()
        assert data["init_parameters"]["host"] == "falkor.test"

        new_store = FalkorDBDocumentStore.from_dict(data)
        assert new_store._host == "falkor.test"
        assert new_store._port == 9999
        assert new_store._graph_name == "my_graph"
        assert new_store._username == "admin"
        assert new_store._password is not None
        assert new_store._node_label == "CustomDoc"
        assert new_store._embedding_dim == 1536
        assert new_store._embedding_field == "vector"
        assert new_store._similarity == "euclidean"
        assert new_store._write_batch_size == 50
        assert new_store._recreate_index is True

    def test_from_dict_without_password(self):
        store_to_serialize = FalkorDBDocumentStore(host="localhost")
        data = store_to_serialize.to_dict()
        new_store = FalkorDBDocumentStore.from_dict(data)
        assert new_store._password is None

    def test_password_not_exposed_in_to_dict(self):
        store = FalkorDBDocumentStore(password=Secret.from_env_var("FALKOR_PWD", strict=False))
        data = store.to_dict()
        assert "type" in data["init_parameters"]["password"]
        assert data["init_parameters"]["password"]["env_vars"] == ["FALKOR_PWD"]

    def test_embedding_retrieval(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore(similarity="cosine")

        mock_res = MagicMock()
        mock_res.result_set = [[{"id": "doc_1", "content": "Text", "score": 0.5}, 0.5]]
        mock_graph.query.return_value = mock_res

        docs = store._embedding_retrieval(query_embedding=[0.1, 0.2], top_k=5, scale_score=True)
        assert len(docs) == 1
        assert docs[0].id == "doc_1"
        assert docs[0].score == (0.5 + 1) / 2  # Cosine scaling

        query = mock_graph.query.call_args[0][0]
        assert "CALL db.idx.vector.queryNodes" in query

    def test_embedding_retrieval_euclidean(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore(similarity="euclidean")

        mock_res = MagicMock()
        mock_res.result_set = [[{"id": "doc_1", "content": "Text"}, 10.0]]
        mock_graph.query.return_value = mock_res

        docs = store._embedding_retrieval(query_embedding=[0.1, 0.2], top_k=5, scale_score=True)
        assert len(docs) == 1
        # Euclidean scaling uses sigmoid: 1 / (1 + exp(-score / 100))
        assert docs[0].score < 1.0

    def test_cypher_retrieval(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_res = MagicMock()
        mock_res.result_set = [[{"id": "doc_1", "content": "Result"}]]
        mock_graph.query.return_value = mock_res

        docs = store._cypher_retrieval("MATCH (d) RETURN d")
        assert len(docs) == 1
        assert docs[0].id == "doc_1"

    def test_cypher_retrieval_error(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        mock_graph.query.side_effect = Exception("Graph error")
        with pytest.raises(Exception, match="Cypher query failed"):
            store._cypher_retrieval("MATCH (d) RETURN d")

    def test_init_verify_connectivity_and_already_initialized(self, mock_falkordb):
        _, _, _mock_graph = mock_falkordb
        store = FalkorDBDocumentStore(verify_connectivity=True)
        assert store._initialized is True

        # Second call should return early
        store._ensure_connected()

    def test_ensure_connected_recreate_index_exception(self, mock_falkordb):
        _mock_client_class, mock_client, _mock_graph = mock_falkordb
        mock_client.delete.side_effect = Exception("Graph not found")

        store = FalkorDBDocumentStore(recreate_index=True)
        store._ensure_connected()
        # Should catch and pass
        mock_client.delete.assert_called_once()

    def test_ensure_schema_exceptions(self, mock_falkordb):
        _mock_client_class, _mock_client, mock_graph = mock_falkordb
        mock_graph.query.side_effect = Exception("Index error")

        # Should gracefully ignore already existing indexes
        store = FalkorDBDocumentStore()
        store._ensure_connected()
        assert mock_graph.query.call_count == 2

    def test_write_batch_document_store_error(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        # Side effect: existing ID ok, but MERGE raises Exception
        def mock_query(cypher, _params=None):
            if "RETURN d.id" in cypher:
                m = MagicMock()
                m.result_set = []
                return m
            elif "MERGE" in cypher:
                msg = "DB Down"
                raise Exception(msg)

        mock_graph.query.side_effect = mock_query

        with pytest.raises(Exception, match="Failed to write documents"):
            store.write_documents([Document(content="Test")])

    def test_embedding_retrieval_with_filters(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore(similarity="cosine")

        mock_res = MagicMock()
        mock_res.result_set = [[{"id": "doc_1", "content": "Text", "score": 0.5}, 0.5]]
        mock_graph.query.return_value = mock_res

        filters = {"field": "type", "operator": "==", "value": "article"}
        store._embedding_retrieval(query_embedding=[0.1, 0.2], top_k=5, filters=filters)

        cypher = mock_graph.query.call_args[0][0]
        assert "WHERE d.type = $p0" in cypher

    def test_filter_documents_or_operator(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        mock_graph.query.return_value = MagicMock(result_set=[])

        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "year", "operator": "==", "value": 2020},
                {"field": "year", "operator": "==", "value": 2021},
            ],
        }
        store.filter_documents(filters)
        cypher = mock_graph.query.call_args[0][0]
        assert "WHERE (d.year = $p0 OR d.year = $p1)" in cypher

    def test_filter_documents_not_operator(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        mock_graph.query.return_value = MagicMock(result_set=[])

        filters = {"operator": "NOT", "conditions": [{"field": "tag", "operator": "==", "value": "old"}]}
        store.filter_documents(filters)
        cypher = mock_graph.query.call_args[0][0]
        assert "WHERE NOT (d.tag = $p0)" in cypher

    def test_filter_documents_in_operators(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        mock_graph.query.return_value = MagicMock(result_set=[])

        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "tag", "operator": "in", "value": ["a", "b"]},
                {"field": "status", "operator": "not in", "value": ["archived"]},
            ],
        }
        store.filter_documents(filters)
        cypher = mock_graph.query.call_args[0][0]
        assert "WHERE (d.tag IN $p0 AND NOT (d.status IN $p1))" in cypher

    @pytest.mark.usefixtures("mock_falkordb")
    def test_filter_documents_unsupported_operator(self):
        store = FalkorDBDocumentStore()
        filters = {"field": "tag", "operator": "contains", "value": "old"}
        with pytest.raises(ValueError, match="Unsupported filter operator: 'contains'"):
            store.filter_documents(filters)

    def test_intra_batch_duplicates(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()
        docs = [
            Document(id="dup", content="1"),
            Document(id="dup", content="2"),
        ]

        # Test SKIP drops the duplicate quietly
        def mock_query(_cypher, _params=None):
            m = MagicMock()
            m.result_set = []
            return m

        mock_graph.query.side_effect = mock_query

        written = store.write_documents(docs, policy=DuplicatePolicy.SKIP)
        assert written == 1

    def test_node_to_document_coverage(self, mock_falkordb):
        _, _, mock_graph = mock_falkordb
        store = FalkorDBDocumentStore()

        # Mock a native FalkorDB node object layout
        class MockNode:
            def __init__(self):
                self.properties = {"id": "real_node_1", "content": "hello"}

        # Mock a random unrecognizable object
        class RandomObj:
            pass

        mock_res = MagicMock()
        mock_res.result_set = [[MockNode()], [RandomObj()]]
        mock_graph.query.return_value = mock_res

        docs = store._cypher_retrieval("MATCH (d) RETURN d")
        assert len(docs) == 2
        assert docs[0].id == "real_node_1"
        assert len(docs[1].id) == 64  # Fallback document ID for empty dictionary is a 64-char hash
