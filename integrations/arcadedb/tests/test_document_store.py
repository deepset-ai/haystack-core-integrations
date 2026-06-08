# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import os
from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)

from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore
from haystack_integrations.document_stores.arcadedb.document_store import (
    _map_literal,
    _map_literal_base,
    _sql_str,
)

ARCADEDB_URL = os.getenv("ARCADEDB_URL", "http://localhost:2480")


def _sample_docs(n: int = 3, dim: int = 4) -> list[Document]:
    docs = []
    for i in range(n):
        docs.append(
            Document(
                content=f"Document number {i}",
                embedding=[float(i)] * dim,
                meta={"category": "test", "priority": i},
            )
        )
    return docs


class TestSqlHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, "NULL"),
            ("hello", "'hello'"),
            ("it's", "'it\\'s'"),
            ("a\\b", "'a\\\\b'"),
            ("", "''"),
        ],
    )
    def test_sql_str(self, value, expected):
        assert _sql_str(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hello", "'hello'"),
            (True, "true"),
            (False, "false"),
            (42, 42),
            (1.5, 1.5),
            (None, "NULL"),
            ([1, 2, 3], [1, 2, 3]),
        ],
    )
    def test_map_literal_base_known_types(self, value, expected):
        assert _map_literal_base(value) == expected

    def test_map_literal_base_fallback_to_str(self):
        dt = datetime.date(2024, 1, 2)
        assert _map_literal_base(dt) == "'2024-01-02'"

    def test_map_literal_empty_dict(self):
        assert _map_literal({}) == "{}"

    def test_map_literal_pairs(self):
        assert _map_literal({"a": 1, "b": "x"}) == '{"a": 1, "b": \'x\'}'


class TestStaticHelpers:
    @pytest.mark.parametrize(
        "rows,expected",
        [
            ([], set()),
            ([{"val": "a"}, {"val": "b"}], {"a", "b"}),
            ([{"val": None}], set()),
            ([{"val": [1, 2, None, 3]}], {"1", "2", "3"}),
            ([{"val": "x"}, {"val": None}, {"val": ["a", "b"]}], {"x", "a", "b"}),
        ],
    )
    def test_extract_distinct_values(self, rows, expected):
        assert ArcadeDBDocumentStore._extract_distinct_values(rows) == expected

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([], "keyword"),
            ([1, 2, 3], "long"),
            ([1.0, 2.0], "double"),
            ([True, False], "boolean"),
            (["a", "b"], "keyword"),
            ([1, "a"], "keyword"),
            ([[1, 2]], "long"),
            ([[True]], "boolean"),
            ([[1.5]], "double"),
            ([["x"]], "keyword"),
        ],
    )
    def test_infer_metadata_field_type(self, values, expected):
        assert ArcadeDBDocumentStore._infer_metadata_field_type(values) == expected


class TestDocumentStoreUnit:
    @pytest.fixture
    def store(self):
        store = ArcadeDBDocumentStore(
            url="http://localhost:2480/",
            database="test",
            embedding_dimension=4,
            create_database=False,
        )
        store._initialized = True
        return store

    def test_to_dict_from_dict(self, store):
        data = store.to_dict()
        restored = ArcadeDBDocumentStore.from_dict(data)
        assert restored._database == store._database
        assert restored._embedding_dimension == store._embedding_dimension
        assert restored._url == store._url

    def test_url_trailing_slash_stripped(self):
        store = ArcadeDBDocumentStore(url="http://host:2480/", database="db")
        assert store._url == "http://host:2480"

    def test_auth_returns_none_when_secrets_unresolved(self, store, monkeypatch):
        monkeypatch.delenv("ARCADEDB_USERNAME", raising=False)
        monkeypatch.delenv("ARCADEDB_PASSWORD", raising=False)

        assert store._auth() is None

    def test_auth_tuple_when_secrets_resolved(self, monkeypatch):
        monkeypatch.setenv("ARCADEDB_USERNAME", "user1")
        monkeypatch.setenv("ARCADEDB_PASSWORD", "pwd1")
        store = ArcadeDBDocumentStore(create_database=False)
        assert store._auth() == ("user1", "pwd1")

    def test_from_dict_with_none_secrets(self):
        data = {
            "type": "haystack_integrations.document_stores.arcadedb.document_store.ArcadeDBDocumentStore",
            "init_parameters": {
                "url": "http://localhost:2480",
                "database": "mydb",
                "username": None,
                "password": None,
                "embedding_dimension": 4,
            },
        }
        restored = ArcadeDBDocumentStore.from_dict(data)
        assert restored._database == "mydb"
        assert restored._username is None
        assert restored._password is None

    def test_command_raises_runtimeerror_on_http_error(self, store):
        mock_resp = MagicMock(status_code=500, text="boom")
        store._session = MagicMock()
        store._session.post.return_value = mock_resp
        with pytest.raises(RuntimeError, match="ArcadeDB command failed"):
            store._command("SELECT 1")

    def test_command_passes_positional_params(self, store):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"result": []}
        store._session = MagicMock()
        store._session.post.return_value = mock_resp
        store._command("SELECT * WHERE id = ?", positional_params=["abc"])
        _, kwargs = store._session.post.call_args
        assert kwargs["json"]["params"] == ["abc"]

    def test_server_command_raises_runtimeerror(self, store):
        mock_resp = MagicMock(status_code=400, text="bad")
        store._session = MagicMock()
        store._session.post.return_value = mock_resp
        with pytest.raises(RuntimeError, match="ArcadeDB server command failed"):
            store._server_command("CREATE DATABASE foo")

    def test_count_documents_returns_zero_when_no_rows(self, store):
        store._command = MagicMock(return_value=[])
        assert store.count_documents() == 0

    @pytest.mark.parametrize(
        "bad_input",
        [
            "not a list",
            [Document(id="1"), "not a doc"],
            [{"dict": "not_doc"}],
        ],
    )
    def test_write_documents_rejects_invalid_input(self, store, bad_input):
        with pytest.raises(ValueError, match="must be a list of Document"):
            store.write_documents(bad_input)

    def test_write_documents_empty_list_returns_zero(self, store):
        store._command = MagicMock()
        assert store.write_documents([]) == 0
        store._command.assert_not_called()

    def test_write_documents_overwrite_inserts_when_update_miss(self, store):
        store._command = MagicMock(side_effect=[[{"count": 0}], []])
        written = store.write_documents(
            [Document(id="x", content="c", embedding=[0.0] * 4)],
            policy=DuplicatePolicy.OVERWRITE,
        )
        assert written == 1
        assert store._command.call_count == 2

    def test_write_documents_skip_returns_existing(self, store):
        store._command = MagicMock(return_value=[{"id": "x"}])
        written = store.write_documents(
            [Document(id="x", content="c", embedding=[0.0] * 4)],
            policy=DuplicatePolicy.SKIP,
        )
        assert written == 0

    def test_delete_documents_empty_list_is_noop(self, store):
        store._command = MagicMock()
        store.delete_documents([])
        store._command.assert_not_called()

    def test_delete_by_filter_empty_filter_raises(self, store):
        with pytest.raises(FilterError, match="non-empty filter"):
            store.delete_by_filter({})

    def test_update_by_filter_empty_filter_raises(self, store):
        with pytest.raises(FilterError, match="non-empty filter"):
            store.update_by_filter({}, {"k": "v"})

    @pytest.mark.parametrize(
        "method,args",
        [
            ("filter_documents", ({"operator": "??", "field": "x", "value": 1},)),
            ("delete_by_filter", ({"operator": "??", "field": "x", "value": 1},)),
            ("update_by_filter", ({"operator": "??", "field": "x", "value": 1}, {"k": "v"})),
            ("count_documents_by_filter", ({"operator": "??", "field": "x", "value": 1},)),
            ("count_unique_metadata_by_filter", ({"operator": "??", "field": "x", "value": 1}, ["k"])),
        ],
    )
    def test_invalid_filter_raises_filter_error(self, store, method, args):
        store._command = MagicMock()
        with pytest.raises(FilterError):
            getattr(store, method)(*args)

    def test_count_documents_by_filter_no_rows(self, store):
        store._command = MagicMock(return_value=[])
        assert store.count_documents_by_filter({"field": "x", "operator": "==", "value": 1}) == 0

    def test_count_unique_metadata_by_filter_empty_fields(self, store):
        store._command = MagicMock()
        assert store.count_unique_metadata_by_filter({}, []) == {}
        store._command.assert_not_called()

    def test_count_unique_metadata_by_filter_with_fields(self, store):
        store._command = MagicMock(return_value=[{"val": "a"}, {"val": "b"}, {"val": "a"}])
        counts = store.count_unique_metadata_by_filter({"field": "meta.x", "operator": "==", "value": 1}, ["meta.cat"])
        assert counts == {"cat": 2}

    def test_get_metadata_fields_info_empty(self, store):
        store._command = MagicMock(return_value=[])
        assert store.get_metadata_fields_info() == {}

    def test_get_metadata_fields_info_infers_types(self, store):
        store._command = MagicMock(
            return_value=[
                {"content": "hi", "meta": {"n": 1, "f": 1.5, "b": True, "s": "x"}},
                {"content": "yo", "meta": {"n": 2, "f": 2.5, "b": False, "s": "y"}},
            ]
        )
        info = store.get_metadata_fields_info()
        assert info["content"] == {"type": "text"}
        assert info["n"] == {"type": "long"}
        assert info["f"] == {"type": "double"}
        assert info["b"] == {"type": "boolean"}
        assert info["s"] == {"type": "keyword"}

    def test_get_metadata_field_min_max_empty_rows(self, store):
        store._command = MagicMock(return_value=[])
        assert store.get_metadata_field_min_max("meta.x") == {"min": None, "max": None}

    def test_embedding_retrieval_returns_empty_when_no_rows(self, store):
        store._command = MagicMock(return_value=[])
        assert store._embedding_retrieval([0.0] * 4) == []

    def test_embedding_retrieval_returns_empty_when_no_neighbors(self, store):
        store._command = MagicMock(return_value=[{"neighbors": None}])
        assert store._embedding_retrieval([0.0] * 4) == []

    def test_embedding_retrieval_post_filters_by_ids(self, store):
        store._command = MagicMock(
            side_effect=[
                [
                    {
                        "neighbors": [
                            {"record": {"id": "a", "content": "A", "meta": {}}, "distance": 0.1},
                            {"record": {"id": "b", "content": "B", "meta": {}}, "distance": 0.2},
                        ]
                    }
                ],
                [{"id": "a"}],
            ]
        )
        docs = store._embedding_retrieval(
            [0.0] * 4,
            filters={"field": "meta.k", "operator": "==", "value": "v"},
        )
        assert [d.id for d in docs] == ["a"]
        assert docs[0].score == pytest.approx(0.9)

    def test_embedding_retrieval_invalid_filter_raises(self, store):
        store._command = MagicMock(return_value=[{"neighbors": [{"record": {"id": "a"}, "distance": 0.0}]}])
        with pytest.raises(FilterError):
            store._embedding_retrieval(
                [0.0] * 4,
                filters={"operator": "??", "field": "x", "value": 1},
            )

    def test_embedding_retrieval_without_filter_returns_all(self, store):
        store._command = MagicMock(
            return_value=[{"neighbors": [{"record": {"id": "a", "content": "A"}, "distance": 0.1}]}]
        )
        docs = store._embedding_retrieval([0.0] * 4)
        assert [d.id for d in docs] == ["a"]


@pytest.mark.skipif(
    not os.environ.get("ARCADEDB_PASSWORD"),
    reason="Set ARCADEDB_PASSWORD (e.g. via repo secret in CI) to run integration tests.",
)
@pytest.mark.integration
class TestArcadeDBDocumentStore(
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Run Haystack DocumentStore mixin tests against ArcadeDBDocumentStore.

    DocumentStoreBaseExtendedTests covers:

        count_documents, delete_documents, filter_documents, write_documents, delete_all_documents, delete_by_filter,
        update_by_filter
    """

    @pytest.fixture
    def document_store(self, document_store: ArcadeDBDocumentStore) -> ArcadeDBDocumentStore:
        """Override to provide ArcadeDB document store from conftest."""
        yield document_store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Compare document lists for tests. Clear score (filter_documents does not set it;
        embedding_retrieval does). Compare embeddings approximately for float round-trip.
        Documents written without embeddings get zero-padded in the store; treat as None for comparison.
        """
        assert len(received) == len(expected)
        received = sorted(received, key=lambda x: x.id)
        expected = sorted(expected, key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            actual = dataclasses.replace(received_doc, score=None)
            if expected_doc.embedding is None:
                actual = dataclasses.replace(actual, embedding=None)
            elif actual.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert actual.embedding == pytest.approx(expected_doc.embedding)
            actual = dataclasses.replace(actual, embedding=None)
            expected_clean = dataclasses.replace(expected_doc, embedding=None)
            assert actual == expected_clean

    def test_write_documents(self, document_store: ArcadeDBDocumentStore):
        """Override mixin: test default write_documents and duplicate fail behaviour."""
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, policy=DuplicatePolicy.FAIL)

    def test_write_overwrite(self, document_store: ArcadeDBDocumentStore):
        """ArcadeDB-specific: overwrite updates content."""
        docs = _sample_docs(1)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        updated = dataclasses.replace(docs[0], content="Updated content")
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 1
        assert all_docs[0].content == "Updated content"

    def test_embedding_retrieval(self, document_store: ArcadeDBDocumentStore):
        """ArcadeDB-specific: vector search via _embedding_retrieval."""
        # Use store's embedding_dimension (768 from conftest); create small test docs
        dim = document_store._embedding_dimension
        docs = [
            Document(
                content=f"Document number {i}",
                embedding=[float(i)] * dim,
                meta={"category": "test", "priority": i},
            )
            for i in range(5)
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._embedding_retrieval(
            query_embedding=[4.0] * dim,
            top_k=3,
        )
        assert len(results) <= 3
        assert results[0].score is not None

    def test_count_documents_by_empty_filter(self, document_store: ArcadeDBDocumentStore):
        """Counts all documents when an empty filter is provided."""
        docs = [
            Document(id="1", content="Doc 1", meta={"category": "news"}),
        ]
        document_store.write_documents(docs)

        count = document_store.count_documents_by_filter({})

        assert count == 1

    def test_count_unique_metadata_by_filter_empty_fields(self, document_store: ArcadeDBDocumentStore):
        """Returns an empty dict when no metadata fields are requested."""
        docs = [
            Document(id="1", content="Doc 1", meta={"category": "news"}),
        ]
        document_store.write_documents(docs)

        counts = document_store.count_unique_metadata_by_filter(
            {"field": "meta.status", "operator": "==", "value": "news"},
            [],
        )

        assert counts == {}

    def test_get_metadata_field_min_max_nonexistent_field(self, document_store: ArcadeDBDocumentStore):
        """Returns None for both min and max when the field does not exist."""
        docs = [Document(id="1", content="Doc 1", meta={"category": "news"})]
        document_store.write_documents(docs)

        result = document_store.get_metadata_field_min_max("nonexistent")

        assert result == {"min": None, "max": None}

    def test_get_metadata_field_unique_values_pagination(self, document_store: ArcadeDBDocumentStore):
        """Respects size limit while total reflects the full unpaginated count."""
        docs = [
            Document(id="1", content="Doc 1", meta={"category": "alpha"}),
            Document(id="2", content="Doc 2", meta={"category": "beta"}),
            Document(id="3", content="Doc 3", meta={"category": "gamma"}),
        ]
        document_store.write_documents(docs)

        values, total = document_store.get_metadata_field_unique_values("category", from_=0, size=2)

        assert len(values) == 2
        assert total == 3

    def test_get_metadata_field_unique_values_case_insensitive(self, document_store: ArcadeDBDocumentStore):
        """Matches values case-insensitively when a search term is provided."""
        docs = [
            Document(id="1", content="Doc 1", meta={"category": "Books"}),
            Document(id="2", content="Doc 2", meta={"category": "books"}),
            Document(id="3", content="Doc 3", meta={"category": "ELECTRONICS"}),
        ]
        document_store.write_documents(docs)

        _, total = document_store.get_metadata_field_unique_values("category", search_term="book")

        assert total == 2

    def test_get_metadata_field_unique_values_no_matches(self, document_store: ArcadeDBDocumentStore):
        """Returns empty results when no metadata values match the search term."""
        docs = [Document(id="1", content="Doc 1", meta={"category": "news"})]
        document_store.write_documents(docs)

        values, total = document_store.get_metadata_field_unique_values("category", search_term="sports")

        assert values == []
        assert total == 0
