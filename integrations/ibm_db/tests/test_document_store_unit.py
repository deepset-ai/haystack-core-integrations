# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests for Db2DocumentStore that never touch a live database.

These tests build a Db2DocumentStore with ``_ensure_table_exists`` patched out and
inject a mocked DB2 connection, so they run fast and cover the Python-side logic
(validation helpers, error handling, async delegation and ``_embedding_retrieval``)
without requiring the FYRE DB2 box.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore
from haystack_integrations.document_stores.ibm_db.document_store import _parse_embedding, _row_to_document


def _make_connection(cursor: MagicMock) -> MagicMock:
    """Build a mock connection whose ``cursor()`` context manager yields ``cursor``."""
    conn = MagicMock()
    cursor_cm = MagicMock()
    cursor_cm.__enter__.return_value = cursor
    cursor_cm.__exit__.return_value = False
    conn.cursor.return_value = cursor_cm
    return conn


@pytest.fixture
def store():
    """A Db2DocumentStore built without connecting to a database."""
    config = Db2ConnectionConfig(database="db", hostname="host", username="u", password="p")
    with patch.object(Db2DocumentStore, "_ensure_table_exists", return_value=None):
        return Db2DocumentStore(
            connection_config=config,
            table_name="unit_docs",
            embedding_dim=4,
            distance_metric="COSINE",
        )


def _attach_connection(store, cursor: MagicMock) -> MagicMock:
    """Give ``store`` a pre-built mock connection so ``_get_connection`` returns it."""
    conn = _make_connection(cursor)
    store._connection = conn
    return conn


# --------------------------------------------------------------------------- #
# _parse_embedding
# --------------------------------------------------------------------------- #
class TestParseEmbedding:
    def test_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_list_of_ints_converted_to_floats(self):
        assert _parse_embedding([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_json_string_list(self):
        assert _parse_embedding("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]

    def test_tuple_is_converted_via_iterable_fallback(self):
        assert _parse_embedding((0.5, 1.5)) == [0.5, 1.5]

    def test_non_numeric_string_returns_none(self):
        # json.loads fails, iterable fallback over characters raises ValueError -> None
        assert _parse_embedding("not-a-vector") is None

    def test_json_string_non_list_falls_through_to_none(self):
        # Valid JSON but not a list -> iterable fallback over its characters raises -> None
        assert _parse_embedding('{"a": 1}') is None

    def test_non_iterable_returns_none(self):
        assert _parse_embedding(object()) is None


# --------------------------------------------------------------------------- #
# _validate_embedding
# --------------------------------------------------------------------------- #
class TestValidateEmbedding:
    def test_none_allowed(self, store):
        # Should not raise
        store._validate_embedding(None, allow_none=True)

    def test_none_not_allowed_raises_value_error(self, store):
        with pytest.raises(ValueError, match="cannot be None"):
            store._validate_embedding(None, allow_none=False)

    def test_non_list_raises_type_error(self, store):
        with pytest.raises(TypeError, match="must be a list"):
            store._validate_embedding("not a list")

    def test_empty_list_raises_value_error(self, store):
        with pytest.raises(ValueError, match="cannot be empty"):
            store._validate_embedding([])

    def test_non_numeric_values_raise_type_error(self, store):
        with pytest.raises(TypeError, match="must be numeric"):
            store._validate_embedding([0.1, "x", 0.3])


# --------------------------------------------------------------------------- #
# _infer_field_type
# --------------------------------------------------------------------------- #
class TestInferFieldType:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "boolean"),
            (10, "integer"),
            (1.5, "real"),
            ("text", "text"),
            ([1, 2], "text"),
            (None, "text"),
        ],
    )
    def test_infer(self, value, expected):
        assert Db2DocumentStore._infer_field_type(value) == expected


# --------------------------------------------------------------------------- #
# write_documents validation / policy dispatch
# --------------------------------------------------------------------------- #
class TestWriteDocumentsUnit:
    def test_empty_list_returns_zero(self, store):
        assert store.write_documents([]) == 0

    def test_invalid_embedding_reraised_with_context(self, store):
        doc = Document(id="1", content="x", embedding=["bad", "values", "here", "!"])
        with pytest.raises(TypeError, match="Invalid embedding for document '1'"):
            store.write_documents([doc])

    def test_insert_documents_translates_duplicate_error(self, store):
        cursor = MagicMock()
        cursor.executemany.side_effect = Exception("SQL0803N duplicate primary key")
        _attach_connection(store, cursor)

        doc = Document(id="1", content="x", embedding=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_insert_documents_reraises_unknown_error(self, store):
        cursor = MagicMock()
        cursor.executemany.side_effect = Exception("some other db error")
        _attach_connection(store, cursor)

        doc = Document(id="1", content="x", embedding=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(Exception, match="some other db error"):
            store.write_documents([doc], policy=DuplicatePolicy.NONE)

    def test_skip_duplicate_documents_counts_inserts(self, store):
        cursor = MagicMock()
        cursor.rowcount = 1
        _attach_connection(store, cursor)

        docs = [Document(id=str(i), content="x", embedding=[0.1, 0.2, 0.3, 0.4]) for i in range(3)]
        assert store.write_documents(docs, policy=DuplicatePolicy.SKIP) == 3

    def test_skip_duplicate_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)

        doc = Document(id="1", content="x", embedding=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(RuntimeError, match="Failed to skip duplicate documents"):
            store.write_documents([doc], policy=DuplicatePolicy.SKIP)

    def test_upsert_documents(self, store):
        cursor = MagicMock()
        _attach_connection(store, cursor)

        docs = [Document(id="1", content="x", embedding=[0.1, 0.2, 0.3, 0.4])]
        assert store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE) == 1

    def test_upsert_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)

        doc = Document(id="1", content="x", embedding=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(RuntimeError, match="Failed to upsert documents"):
            store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)


# --------------------------------------------------------------------------- #
# count / delete / update / filter error + early-return branches
# --------------------------------------------------------------------------- #
class TestQueryBranches:
    def test_count_documents_returns_zero_on_empty_result(self, store):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        _attach_connection(store, cursor)
        assert store.count_documents() == 0

    def test_count_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to count documents"):
            store.count_documents()

    def test_count_documents_by_filter_no_filters_delegates(self, store):
        store.count_documents = MagicMock(return_value=7)
        assert store.count_documents_by_filter(None) == 7

    def test_count_documents_by_filter_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to count documents by filter"):
            store.count_documents_by_filter({"operator": "==", "field": "meta.a", "value": 1})

    def test_filter_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to filter documents"):
            store.filter_documents()

    def test_build_where_clause_invalid_filter_raises_filter_error(self, store):
        with pytest.raises(FilterError):
            store._build_where_clause({"operator": "bogus", "field": "x", "value": 1})

    def test_delete_documents_empty_is_noop(self, store):
        # No connection attached; should return before touching the DB.
        store.delete_documents([])

    def test_delete_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to delete documents"):
            store.delete_documents(["1", "2"])

    def test_delete_by_filter_no_filters_returns_zero(self, store):
        assert store.delete_by_filter(None) == 0

    def test_delete_by_filter_counts_rows(self, store):
        cursor = MagicMock()
        cursor.rowcount = 5
        _attach_connection(store, cursor)
        assert store.delete_by_filter({"operator": "==", "field": "meta.a", "value": 1}) == 5

    def test_delete_by_filter_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to delete documents by filter"):
            store.delete_by_filter({"operator": "==", "field": "meta.a", "value": 1})

    def test_delete_all_documents_deletes_rows(self, store):
        cursor = MagicMock()
        cursor.fetchone.return_value = (4,)
        _attach_connection(store, cursor)
        assert store.delete_all_documents(recreate_index=False) == 4

    def test_delete_all_documents_recreate(self, store):
        cursor = MagicMock()
        cursor.fetchone.return_value = (2,)
        _attach_connection(store, cursor)
        with patch.object(store, "_ensure_table_exists", return_value=None) as ensure:
            assert store.delete_all_documents(recreate_index=True) == 2
        ensure.assert_called_once_with(recreate=True)

    def test_delete_all_documents_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to delete all documents"):
            store.delete_all_documents()

    def test_update_by_filter_requires_meta(self, store):
        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            store.update_by_filter({"operator": "==", "field": "meta.a", "value": 1}, meta={})

    def test_update_by_filter_no_filters_returns_zero(self, store):
        assert store.update_by_filter(None, meta={"a": 1}) == 0

    def test_update_by_filter_merges_and_counts(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = [("doc1", '{"a": 1}'), ("doc2", None)]
        _attach_connection(store, cursor)
        updated = store.update_by_filter({"operator": "==", "field": "meta.a", "value": 1}, meta={"b": 2})
        assert updated == 2

    def test_update_by_filter_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to update documents by filter"):
            store.update_by_filter({"operator": "==", "field": "meta.a", "value": 1}, meta={"b": 2})


# --------------------------------------------------------------------------- #
# metadata helpers
# --------------------------------------------------------------------------- #
class TestMetadataHelpers:
    def test_get_unique_values_dedup_and_parse(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = [('"a"',), ('"a"',), ('"b"',), (None,), ("plain",)]
        _attach_connection(store, cursor)
        values = store.get_metadata_field_unique_values("meta.category")
        assert values == ["a", "b", "plain"]

    def test_get_unique_values_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to get unique values"):
            store.get_metadata_field_unique_values("meta.category")

    def test_min_max_numeric(self, store):
        cursor = MagicMock()
        cursor.fetchone.return_value = (1.0, 9.0)
        _attach_connection(store, cursor)
        assert store.get_metadata_field_min_max("meta.age") == {"min": 1.0, "max": 9.0}

    def test_min_max_falls_back_to_lexicographic(self, store):
        cursor = MagicMock()
        # First execute (numeric cast) raises; second (lexicographic) succeeds.
        cursor.execute.side_effect = [Exception("cast failed"), None]
        cursor.fetchone.return_value = ("apple", "pear")
        _attach_connection(store, cursor)
        assert store.get_metadata_field_min_max("meta.name") == {"min": "apple", "max": "pear"}

    def test_min_max_none_when_no_rows(self, store):
        cursor = MagicMock()
        cursor.fetchone.return_value = (None, None)
        _attach_connection(store, cursor)
        assert store.get_metadata_field_min_max("meta.age") == {"min": None, "max": None}

    def test_get_metadata_fields_info(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ('{"age": 5, "name": "x", "flag": true, "ratio": 1.5, "empty": null}',),
            (None,),
            ("[1, 2, 3]",),  # not a dict -> skipped
        ]
        _attach_connection(store, cursor)
        info = store.get_metadata_fields_info()
        assert info["age"] == {"type": "integer"}
        assert info["name"] == {"type": "text"}
        assert info["flag"] == {"type": "boolean"}
        assert info["ratio"] == {"type": "real"}
        assert info["empty"] == {"type": "text"}

    def test_get_metadata_fields_info_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to get metadata fields info"):
            store.get_metadata_fields_info()

    def test_count_unique_metadata_empty_fields(self, store):
        assert store.count_unique_metadata_by_filter(metadata_fields=None) == {}

    def test_count_unique_metadata_with_filter(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = [('"x"',), ('"x"',), ('"y"',), (None,)]
        _attach_connection(store, cursor)
        result = store.count_unique_metadata_by_filter(
            filters={"operator": "==", "field": "meta.a", "value": 1},
            metadata_fields=["meta.category"],
        )
        assert result == {"meta.category": 2}

    def test_count_unique_metadata_error_wrapped(self, store):
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("boom")
        _attach_connection(store, cursor)
        with pytest.raises(RuntimeError, match="Failed to count unique metadata"):
            store.count_unique_metadata_by_filter(metadata_fields=["meta.category"])


# --------------------------------------------------------------------------- #
# _embedding_retrieval
# --------------------------------------------------------------------------- #
class TestEmbeddingRetrieval:
    def test_returns_documents_with_scores(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ("doc1", "content one", '{"a": 1}', [0.1, 0.2, 0.3, 0.4], 0.25),
            ("doc2", "content two", None, None, 0.75),
        ]
        _attach_connection(store, cursor)

        docs = store._embedding_retrieval([0.1, 0.2, 0.3, 0.4], top_k=2)
        assert [d.id for d in docs] == ["doc1", "doc2"]
        assert docs[0].meta == {"a": 1}
        assert docs[0].embedding == [0.1, 0.2, 0.3, 0.4]
        assert docs[0].score == 0.25
        assert docs[1].meta == {}
        assert docs[1].embedding is None

    def test_with_filters_appends_null_check(self, store):
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn = _attach_connection(store, cursor)

        store._embedding_retrieval(
            [0.1, 0.2, 0.3, 0.4],
            filters={"operator": "==", "field": "meta.a", "value": 1},
            top_k=5,
        )
        conn.cursor.assert_called()
        executed_sql = cursor.execute.call_args[0][0]
        assert "embedding IS NOT NULL" in executed_sql
        assert "WHERE" in executed_sql

    def test_division_by_zero_returns_empty(self, store):
        cursor = MagicMock()
        cursor.fetchall.side_effect = Exception("SQL0801N Division by zero")
        _attach_connection(store, cursor)
        assert store._embedding_retrieval([0.0, 0.0, 0.0, 0.0], top_k=3) == []

    def test_unrelated_error_reraised(self, store):
        cursor = MagicMock()
        cursor.fetchall.side_effect = Exception("some fatal error")
        _attach_connection(store, cursor)
        with pytest.raises(Exception, match="some fatal error"):
            store._embedding_retrieval([0.1, 0.2, 0.3, 0.4], top_k=3)

    def test_invalid_query_embedding_raises(self, store):
        with pytest.raises(ValueError, match="cannot be None"):
            store._embedding_retrieval(None, top_k=3)


# --------------------------------------------------------------------------- #
# _get_connection: SSL, schema, extra options
# --------------------------------------------------------------------------- #
class TestGetConnection:
    def test_builds_ssl_and_schema_connection(self):
        config = Db2ConnectionConfig(
            database="db",
            hostname="host",
            username="u",
            password="p",
            schema="MYSCHEMA",
            use_ssl=True,
            ssl_certificate="/path/to/cert.arm",
            connection_options={"CURRENTSCHEMA": "X"},
        )
        with patch.object(Db2DocumentStore, "_ensure_table_exists", return_value=None):
            store = Db2DocumentStore(connection_config=config, table_name="t", embedding_dim=4)

        schema_cursor = MagicMock()
        conn = _make_connection(schema_cursor)

        with patch(
            "haystack_integrations.document_stores.ibm_db.document_store.ibm_db_dbi.pconnect",
            return_value=conn,
        ) as pconnect:
            result = store._get_connection()

        assert result is conn
        dsn = pconnect.call_args.kwargs["dsn"]
        assert "SECURITY=SSL" in dsn
        assert "SSLServerCertificate=/path/to/cert.arm" in dsn
        schema_cursor.execute.assert_called_once_with("SET SCHEMA MYSCHEMA")

    def test_schema_failure_raises_runtime_error(self):
        config = Db2ConnectionConfig(database="db", hostname="host", username="u", password="p", schema="BAD")
        with patch.object(Db2DocumentStore, "_ensure_table_exists", return_value=None):
            store = Db2DocumentStore(connection_config=config, table_name="t", embedding_dim=4)

        schema_cursor = MagicMock()
        schema_cursor.execute.side_effect = Exception("no such schema")
        conn = _make_connection(schema_cursor)

        with patch(
            "haystack_integrations.document_stores.ibm_db.document_store.ibm_db_dbi.pconnect",
            return_value=conn,
        ):
            with pytest.raises(RuntimeError, match="Failed to set schema BAD"):
                store._get_connection()


# --------------------------------------------------------------------------- #
# async wrappers delegate to their sync counterparts
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestAsyncWrappers:
    async def test_count_documents_async(self, store):
        store.count_documents = MagicMock(return_value=3)
        assert await store.count_documents_async() == 3

    async def test_count_documents_by_filter_async(self, store):
        store.count_documents_by_filter = MagicMock(return_value=2)
        assert await store.count_documents_by_filter_async({"f": 1}) == 2

    async def test_write_documents_async(self, store):
        store.write_documents = MagicMock(return_value=5)
        assert await store.write_documents_async([], DuplicatePolicy.SKIP) == 5

    async def test_filter_documents_async(self, store):
        sentinel = [Document(id="1")]
        store.filter_documents = MagicMock(return_value=sentinel)
        assert await store.filter_documents_async() is sentinel

    async def test_delete_documents_async(self, store):
        store.delete_documents = MagicMock(return_value=None)
        await store.delete_documents_async(["1"])
        store.delete_documents.assert_called_once_with(["1"])

    async def test_delete_by_filter_async(self, store):
        store.delete_by_filter = MagicMock(return_value=4)
        assert await store.delete_by_filter_async({"f": 1}) == 4

    async def test_delete_all_documents_async(self, store):
        store.delete_all_documents = MagicMock(return_value=9)
        assert await store.delete_all_documents_async(True) == 9

    async def test_update_by_filter_async(self, store):
        store.update_by_filter = MagicMock(return_value=1)
        assert await store.update_by_filter_async({"f": 1}, {"a": 1}) == 1

    async def test_get_metadata_field_unique_values_async(self, store):
        store.get_metadata_field_unique_values = MagicMock(return_value=["a"])
        assert await store.get_metadata_field_unique_values_async("meta.x") == ["a"]

    async def test_get_metadata_field_min_max_async(self, store):
        store.get_metadata_field_min_max = MagicMock(return_value={"min": 0, "max": 1})
        assert await store.get_metadata_field_min_max_async("meta.x") == {"min": 0, "max": 1}

    async def test_get_metadata_fields_info_async(self, store):
        store.get_metadata_fields_info = MagicMock(return_value={"x": {"type": "text"}})
        assert await store.get_metadata_fields_info_async() == {"x": {"type": "text"}}

    async def test_count_unique_metadata_by_filter_async(self, store):
        store.count_unique_metadata_by_filter = MagicMock(return_value={"x": 1})
        assert await store.count_unique_metadata_by_filter_async(None, ["meta.x"]) == {"x": 1}

    async def test_embedding_retrieval_async(self, store):
        sentinel = [Document(id="1")]
        store._embedding_retrieval = MagicMock(return_value=sentinel)
        result = await store._embedding_retrieval_async([0.1, 0.2, 0.3, 0.4], filters=None, top_k=3)
        assert result is sentinel


# --------------------------------------------------------------------------- #
# _row_to_document via non-string embedding
# --------------------------------------------------------------------------- #
def test_row_to_document_parses_string_embedding():
    row = ("id", "content", '{"k": "v"}', "[0.1, 0.2]")
    doc = _row_to_document(row)
    assert doc.embedding == [0.1, 0.2]
    assert doc.meta == {"k": "v"}
