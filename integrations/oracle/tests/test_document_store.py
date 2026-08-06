# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from unittest.mock import MagicMock

import oracledb as _oracledb
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    DocumentStoreBaseTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
)
from haystack.testing.document_store_async import (
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterAsyncTest,
)

from haystack_integrations.document_stores.oracle import OracleDocumentStore


def _doc(doc_id: str, content: str = "hello", meta: dict | None = None, embedding: list[float] | None = None):
    """Integration test document builder — doc_id is required, no embedding default."""
    return Document(id=doc_id, content=content, meta=meta or {}, embedding=embedding)


def _uid(suffix: str = "") -> str:
    """Return a 32-char uppercase hex ID, optionally tagged."""
    base = uuid.uuid4().hex.upper()[:28]
    return f"{base}{suffix.upper():>4}"[:32]


def _lob(value: str) -> MagicMock:
    """Mock for an oracledb Large Object (LOB)"""
    lob = MagicMock()
    lob.read.return_value = value
    return lob


class TestOracleDocumentStoreUnit:
    @staticmethod
    def _mock_doc(content="hello", embedding=None, doc_id="AABB" * 8):
        """Lightweight document builder for mock-based tests."""
        return Document(id=doc_id, content=content, meta={"k": "v"}, embedding=embedding)

    @pytest.mark.parametrize(
        ("table_name", "embedding_dim", "match"),
        [
            pytest.param("1invalid", 4, "Invalid table_name", id="table_name_starts_with_digit"),
            pytest.param("bad-name", 4, "Invalid table_name", id="table_name_illegal_char"),
            pytest.param("valid", 0, "embedding_dim must be a positive integer", id="embedding_dim_zero"),
            pytest.param("valid", -1, "embedding_dim must be a positive integer", id="embedding_dim_negative"),
        ],
    )
    def test_init_rejects_invalid_arguments(self, make_store, table_name, embedding_dim, match):
        with pytest.raises(ValueError, match=match):
            make_store(table_name, embedding_dim)

    def test_init_opens_no_connection(self, make_store, mock_pool):  # noqa: ARG002
        store = make_store("lazy_docs")
        assert store._pool is None
        assert store._setup_done is False

    def test_setup_is_deferred_and_runs_once(self, make_store, mock_pool):
        store = make_store("lazy_docs")
        store.count_documents()
        store.count_documents()
        assert store._pool is not None
        assert store._setup_done is True
        _, _, cursor = mock_pool
        create_calls = [c for c in cursor.execute.call_args_list if "CREATE TABLE IF NOT EXISTS" in c[0][0]]
        assert len(create_calls) == 1

    def test_setup_failure_resets_flag(self, make_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.execute.side_effect = RuntimeError("boom")
        store = make_store("lazy_docs")
        with pytest.raises(RuntimeError):
            store.count_documents()
        assert store._setup_done is False

    def test_write_documents_none_policy_calls_insert(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.write_documents([self._mock_doc()], policy=DuplicatePolicy.NONE)
        cursor.executemany.assert_called_once()
        sql = cursor.executemany.call_args[0][0]
        assert "INSERT INTO" in sql
        assert ":doc_id" in sql

    def test_write_documents_none_policy_duplicate_raises(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.executemany.side_effect = _oracledb.IntegrityError("ORA-00001")
        with pytest.raises(DuplicateDocumentError):
            patched_store.write_documents([self._mock_doc()], policy=DuplicatePolicy.NONE)

    def test_write_documents_skip_policy_uses_merge_not_matched(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.write_documents([self._mock_doc()], policy=DuplicatePolicy.SKIP)
        sql = cursor.executemany.call_args[0][0]
        assert "MERGE INTO" in sql
        assert "WHEN NOT MATCHED" in sql
        assert "WHEN MATCHED" not in sql

    def test_write_documents_overwrite_policy_uses_full_merge(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.write_documents([self._mock_doc()], policy=DuplicatePolicy.OVERWRITE)
        sql = cursor.executemany.call_args[0][0]
        assert "MERGE INTO" in sql
        assert "WHEN MATCHED" in sql
        assert "WHEN NOT MATCHED" in sql

    def test_write_documents_returns_count(self, patched_store, mock_pool):  # noqa: ARG002
        count = patched_store.write_documents(
            [self._mock_doc(), self._mock_doc(doc_id="CCDD" * 8)], policy=DuplicatePolicy.NONE
        )
        assert count == 2

    def test_write_documents_empty_list_no_db_call(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        assert patched_store.write_documents([], policy=DuplicatePolicy.NONE) == 0
        cursor.executemany.assert_not_called()

    def test_filter_documents_no_filter_fetches_all(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.fetchall.return_value = [
            ("AABB" * 8, "hello", '{"k": "v"}'),
            ("CCDD" * 8, "world", "{}"),
        ]
        docs = patched_store.filter_documents()
        assert len(docs) == 2
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_filter_documents_equality_filter_produces_correct_sql(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.fetchall.return_value = []
        patched_store.filter_documents(filters={"field": "meta.author", "operator": "==", "value": "Alice"})
        sql, params = cursor.execute.call_args[0]
        assert "JSON_VALUE(metadata, '$.author') = :p0" in sql
        assert params["p0"] == "Alice"

    def test_filter_documents_and_filter(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.fetchall.return_value = []
        patched_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.year", "operator": ">", "value": 2020},
                ],
            }
        )
        sql, params = cursor.execute.call_args[0]
        assert "AND" in sql
        assert len(params) == 2

    def test_delete_documents_builds_correct_sql(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.delete_documents(["AABB" * 8, "CCDD" * 8])
        sql = cursor.execute.call_args[0][0]
        assert "DELETE FROM" in sql
        assert "IN (:p0, :p1)" in sql

    def test_delete_documents_empty_list_is_noop(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.delete_documents([])
        cursor.execute.assert_not_called()

    def test_count_documents_returns_value(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.fetchone.return_value = (42,)
        assert patched_store.count_documents() == 42

    def test_get_metadata_field_unique_values_search_term_filters_value_only(self, patched_store, mock_pool):
        """search_term must filter on the metadata field's own JSON value, not on document text."""
        _, _, cursor = mock_pool
        cursor.fetchone.return_value = (1,)
        cursor.fetchall.return_value = [("bar",)]

        patched_store.get_metadata_field_unique_values("category", search_term="bar")

        count_sql, count_params = cursor.execute.call_args_list[0][0]
        vals_sql, vals_params = cursor.execute.call_args_list[1][0]

        for sql in (count_sql, vals_sql):
            assert "UPPER(JSON_VALUE(metadata, '$.category')) LIKE UPPER(:search)" in sql

        assert count_params["search"] == "%bar%"
        assert vals_params["search"] == "%bar%"

    def test_get_metadata_field_unique_values_parses_json_scalar_types(self, patched_store, mock_pool):
        """Numeric/boolean JSON_VALUE results (returned by Oracle as text) are parsed back to their
        original type; plain strings (not valid JSON when unquoted) are left as-is."""
        _, _, cursor = mock_pool
        cursor.fetchone.return_value = (3,)
        cursor.fetchall.return_value = [("1",), ("true",), ("bar",)]

        values, total = patched_store.get_metadata_field_unique_values("priority")

        assert values == [1, True, "bar"]
        assert total == 3

    def test_delete_table_executes_drop_and_index_sql(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.delete_table()
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("DROP TABLE" in sql and "PURGE" in sql for sql in executed)
        assert any("DBMS_SEARCH.DROP_INDEX" in sql for sql in executed)

    def test_keyword_retrieval_builds_contains_query(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        cursor.fetchall.return_value = [("AABB" * 8, "hello", '{"k": "v"}', 1.5)]
        docs = patched_store._keyword_retrieval("hello world", top_k=5)
        sql, params = cursor.execute.call_args[0]
        assert "CONTAINS(DATA, :query, 1)" in sql
        assert "SCORE(1)" in sql
        assert params["query"] == "hello world"
        assert params["top_k"] == 5
        assert docs[0].score == 1.5

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda s: s._embedding_retrieval([0.1, 0.2, 0.3, 0.4]), id="embedding_retrieval"),
            pytest.param(lambda s: s._keyword_retrieval("query"), id="keyword_retrieval"),
            pytest.param(lambda s: s.delete_table(), id="delete_table"),
        ],
    )
    def test_query_methods_wrap_database_errors(self, patched_store, mock_pool, operation):
        _, _, cursor = mock_pool
        cursor.execute.side_effect = _oracledb.DatabaseError("ORA-00942")
        with pytest.raises(DocumentStoreError):
            operation(patched_store)

    @pytest.mark.parametrize(
        ("text_raw", "metadata_raw", "expected_content", "expected_meta"),
        [
            pytest.param("plain", '{"k": "v"}', "plain", {"k": "v"}, id="str_text_and_json_metadata"),
            pytest.param(_lob("from_lob"), _lob('{"k": "v"}'), "from_lob", {"k": "v"}, id="lob_text_and_metadata"),
            pytest.param("plain", {"k": "v"}, "plain", {"k": "v"}, id="dict_metadata"),
            pytest.param("plain", None, "plain", {}, id="unsupported_metadata_type"),
        ],
    )
    def test_row_to_document_reads_lob_and_maps_metadata(self, text_raw, metadata_raw, expected_content, expected_meta):
        doc = OracleDocumentStore._row_to_document(("id1", text_raw, metadata_raw))
        assert doc.content == expected_content
        assert doc.meta == expected_meta

    def test_create_hnsw_index_sql(self, patched_store, mock_pool):
        _, _, cursor = mock_pool
        patched_store.create_hnsw_index()
        sql = cursor.execute.call_args[0][0]
        assert "CREATE VECTOR INDEX" in sql
        assert "HNSW" in sql
        assert str(patched_store.hnsw_neighbors) in sql
        assert str(patched_store.hnsw_ef_construction) in sql

    def test_to_dict_does_not_expose_plain_password(self, patched_store):
        d = patched_store.to_dict()
        pw = d["init_parameters"]["connection_config"]["password"]
        assert isinstance(pw, dict)
        assert pw.get("type") == "env_var"

    def test_from_dict_roundtrip(self, patched_store):
        d = patched_store.to_dict()
        restored = OracleDocumentStore.from_dict(d)
        assert restored.table_name == patched_store.table_name
        assert restored.embedding_dim == patched_store.embedding_dim
        assert restored.distance_metric == patched_store.distance_metric

    def test_close(self, patched_store):
        pool = MagicMock()
        patched_store._pool = pool
        patched_store.close()
        pool.close.assert_called_once()
        assert patched_store._pool is None
        patched_store.close()
        pool.close.assert_called_once()

    def test_close_is_exception_safe(self, patched_store):
        pool = MagicMock()
        pool.close.side_effect = RuntimeError("boom")
        patched_store._pool = pool
        patched_store.close()
        assert patched_store._pool is None


@pytest.mark.integration
class TestOracleDocumentStore(
    DocumentStoreBaseTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
):
    @pytest.fixture
    def document_store(self, make_store):
        """768-dim store — overrides the mixin's NotImplementedError stub."""
        table = f"hs_sync_{uuid.uuid4().hex[:8]}"
        s = make_store(table, 768)
        yield s
        with s._get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE {table} PURGE")
            conn.commit()

    # Mixin override
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        # filter_documents does not SELECT the embedding column — ignore it when comparing
        assert len(received) == len(expected)
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)
        for r, e in zip(received_sorted, expected_sorted, strict=True):
            assert r.id == e.id
            assert r.content == e.content
            assert r.meta == e.meta

    # Mixin override
    def test_write_documents(self, document_store: OracleDocumentStore) -> None:
        # Default policy is NONE — a second write of the same doc raises DuplicateDocumentError
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc])
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    # test_comparison_equal_with_none     → IS NULL
    # test_comparison_not_equal_with_none → IS NOT NULL
    # test_comparison_not_equal           → col != x OR IS NULL
    # test_comparison_not_in              → IS NULL OR NOT IN
    @pytest.mark.skip(
        reason="Oracle NULL propagation in NOT(...) cannot match Python 'not (None == x) is True' semantics"
    )
    def test_not_operator(self, document_store, filterable_docs): ...

    def test_filter_documents_not_operator(self, document_store):
        # Scoped to two fresh docs — no NULL-valued rows, so NOT works correctly.
        tag = _uid("I3")[:8]
        document_store.write_documents(
            [
                _doc(_uid("I301"), meta={"tag": tag, "lang": "en"}),
                _doc(_uid("I302"), meta={"tag": tag, "lang": "de"}),
            ]
        )
        results = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.tag", "operator": "==", "value": tag},
                    {
                        "operator": "NOT",
                        "conditions": [{"field": "meta.lang", "operator": "==", "value": "de"}],
                    },
                ],
            }
        )
        assert len(results) == 1
        assert results[0].meta["lang"] == "en"

    def test_embedding_retrieval_order(self, embedding_store):
        tag = _uid("J")[:8]
        embedding_store.write_documents(
            [
                _doc(_uid("J001"), content="near", meta={"tag": tag}, embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc(_uid("J002"), content="far", meta={"tag": tag}, embedding=[0.0, 0.0, 0.0, 1.0]),
                _doc(_uid("J003"), content="medium", meta={"tag": tag}, embedding=[0.7, 0.7, 0.0, 0.0]),
            ]
        )
        results = embedding_store._embedding_retrieval(
            [1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.tag", "operator": "==", "value": tag},
            top_k=3,
        )
        assert results[0].content == "near"
        assert len(results) == 3

    def test_embedding_retrieval_top_k(self, embedding_store):
        tag = _uid("K")[:8]
        embedding_store.write_documents(
            [_doc(_uid(f"K{i:03}"), meta={"tag": tag}, embedding=[1.0, 0.0, 0.0, 0.0]) for i in range(5)]
        )
        results = embedding_store._embedding_retrieval(
            [1.0, 0.0, 0.0, 0.0],
            filters={"field": "meta.tag", "operator": "==", "value": tag},
            top_k=3,
        )
        assert len(results) == 3

    def test_embedding_retrieval_with_filter(self, embedding_store):
        tag = _uid("L")[:8]
        embedding_store.write_documents(
            [
                _doc(_uid("L001"), content="en", meta={"tag": tag, "lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]),
                _doc(_uid("L002"), content="de", meta={"tag": tag, "lang": "de"}, embedding=[1.0, 0.0, 0.0, 0.0]),
            ]
        )
        results = embedding_store._embedding_retrieval(
            [1.0, 0.0, 0.0, 0.0],
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.tag", "operator": "==", "value": tag},
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                ],
            },
            top_k=10,
        )
        assert all(d.meta["lang"] == "en" for d in results)
        assert len(results) == 1

    def test_create_table_idempotent(self, document_store):
        """Calling _ensure_table() a second time must not raise."""
        document_store._ensure_table()

    def test_close_and_reopen(self, document_store):
        document_store.count_documents()
        assert document_store._pool is not None
        document_store.close()
        assert document_store._pool is None
        assert document_store.count_documents() == 0
        assert document_store._pool is not None

    def test_get_metadata_field_unique_values_search_term_matches_value_not_content(self, document_store):
        """search_term filters on the metadata field's own value; document content is not considered."""
        document_store.write_documents(
            [
                _doc(_uid("Q001"), content="this text mentions apple", meta={"category": "dessert"}),
                _doc(_uid("Q002"), content="unrelated content", meta={"category": "apple-tart"}),
            ]
        )
        values, total = document_store.get_metadata_field_unique_values("category", search_term="apple")
        # Q001: content contains "apple" but its category value ("dessert") does not -> excluded.
        # Q002: category value ("apple-tart") contains "apple", content does not -> still included.
        assert values == ["apple-tart"]
        assert total == 1

    def test_get_metadata_field_unique_values_search_term_case_insensitive(self, document_store):
        document_store.write_documents(
            [
                _doc(_uid("Q003"), content="n/a", meta={"category": "Apple-Tart"}),
            ]
        )
        values, total = document_store.get_metadata_field_unique_values("category", search_term="APPLE")
        assert values == ["Apple-Tart"]
        assert total == 1

    def test_get_metadata_field_unique_values_preserves_non_string_types(self, document_store):
        """Non-string metadata values (e.g. ints) are returned in their original type, not stringified."""
        document_store.write_documents(
            [
                _doc(_uid("P001"), content="one", meta={"priority": 1}),
                _doc(_uid("P002"), content="two", meta={"priority": 2}),
                _doc(_uid("P003"), content="three", meta={"priority": 1}),
            ]
        )
        values, total = document_store.get_metadata_field_unique_values("priority")
        assert set(values) == {1, 2}
        assert total == 2

    def test_get_metadata_field_unique_values_with_filters(self, document_store):
        document_store.write_documents(
            [
                _doc(_uid("R001"), meta={"category": "A", "status": "active"}),
                _doc(_uid("R002"), meta={"category": "B", "status": "active"}),
                _doc(_uid("R003"), meta={"category": "C", "status": "inactive"}),
            ]
        )

        filters = {"field": "meta.status", "operator": "==", "value": "active"}
        values, total = document_store.get_metadata_field_unique_values("category", filters=filters)
        assert set(values) == {"A", "B"}
        assert total == 2


@pytest.mark.integration
class TestOracleDocumentStoreAsync(
    FilterableDocsFixtureMixin,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    UpdateByFilterAsyncTest,
):
    """Async API surface tests."""

    @pytest.mark.asyncio
    async def test_async_write_and_count(self, embedding_store):
        doc = _doc(_uid("M001"))
        await embedding_store.write_documents_async([doc])
        count = await embedding_store.count_documents_async()
        assert count >= 1

    @pytest.mark.asyncio
    async def test_async_filter_documents(self, embedding_store):
        tag = _uid("N")[:8]
        doc = _doc(_uid("N001"), meta={"tag": tag})
        await embedding_store.write_documents_async([doc])
        results = await embedding_store.filter_documents_async(
            filters={"field": "meta.tag", "operator": "==", "value": tag}
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_async_delete_documents(self, embedding_store):
        doc_id = _uid("O001")
        await embedding_store.write_documents_async([_doc(doc_id)])
        await embedding_store.delete_documents_async([doc_id])
        results = await embedding_store.filter_documents_async(
            filters={"field": "id", "operator": "==", "value": doc_id}
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_async_write_and_retrieve(self, embedding_store):
        doc_id = _uid("P001")
        await embedding_store.write_documents_async([_doc(doc_id, embedding=[0.5, 0.5, 0.0, 0.0])])
        results = await embedding_store._embedding_retrieval_async([0.5, 0.5, 0.0, 0.0], top_k=1)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_with_filters_async(self, embedding_store):
        await embedding_store.write_documents_async(
            [
                _doc(_uid("S001"), meta={"category": "A", "status": "active"}),
                _doc(_uid("S002"), meta={"category": "B", "status": "active"}),
                _doc(_uid("S003"), meta={"category": "C", "status": "inactive"}),
            ]
        )

        filters = {"field": "meta.status", "operator": "==", "value": "active"}
        values, total = await embedding_store.get_metadata_field_unique_values_async("category", filters=filters)
        assert set(values) == {"A", "B"}
        assert total == 2
