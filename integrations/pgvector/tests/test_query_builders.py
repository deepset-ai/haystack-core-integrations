# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psycopg
import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from psycopg import AsyncCursor, Cursor, Error, IntegrityError
from psycopg.adapt import AdaptersMap, Transformer
from psycopg.sql import SQL, Composed

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

EQUALS_BOOK = {"field": "meta.kind", "operator": "==", "value": "book"}

STORE_MODULE = "haystack_integrations.document_stores.pgvector.document_store"


def _render(query: Composed | SQL) -> str:
    """Render a psycopg query object to a plain SQL string for assertions."""
    return query.as_string(Transformer())


def _record(**overrides) -> dict:
    """A row as the store reads it back, with every column the converter expects."""
    record = {
        "id": "doc-1",
        "content": "alpha",
        "meta": {},
        "embedding": None,
        "blob_data": None,
        "blob_meta": None,
        "blob_mime_type": None,
    }
    record.update(overrides)
    return record


def _mock_cursor(spec=Cursor) -> Mock:
    cursor = MagicMock(spec=spec)
    # `_execute_sql` renders the query against the cursor before running it, and psycopg
    # needs a real adapters map to do that.
    cursor.adapters = AdaptersMap(psycopg.adapters)
    cursor.connection = None
    return cursor


@pytest.fixture
def cursor(mock_store_with_mock_connection) -> Mock:
    """A cursor whose results the store reads through, with no database behind it."""
    cursor = _mock_cursor()
    cursor.execute.return_value = cursor
    # `write_documents` walks the RETURNING id result sets until nextset() runs out.
    cursor.fetchone.return_value = None
    cursor.nextset.return_value = None
    store = mock_store_with_mock_connection
    store._cursor = cursor
    store._dict_cursor = cursor
    store._table_initialized = True
    return cursor


@pytest.fixture
def async_cursor(mock_store_with_mock_async_connection) -> Mock:
    cursor = _mock_cursor(spec=AsyncCursor)
    cursor.execute = AsyncMock(return_value=cursor)
    cursor.executemany = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.nextset = AsyncMock(return_value=None)
    # `write_documents_async` counts the inserted ids by iterating the cursor.
    cursor.__aiter__.return_value = iter(())
    store = mock_store_with_mock_async_connection
    store._async_cursor = cursor
    store._async_dict_cursor = cursor
    store._table_initialized = True
    return cursor


@pytest.fixture
def async_store(mock_store_with_mock_async_connection, async_cursor):  # noqa: ARG001
    """An async store whose db setup is already done, so no connection is opened."""
    with patch.object(PgvectorDocumentStore, "_ensure_db_setup_async", new=AsyncMock()):
        yield mock_store_with_mock_async_connection


class TestTableCreationQueries:
    def test_names_the_configured_schema_and_table(self, mock_store):
        _, create_table, _, _ = mock_store._build_table_creation_queries()

        rendered = _render(create_table)
        assert '"public"."haystack"' in rendered
        assert "vector(768)" in rendered


class TestHnswQueries:
    def test_sets_ef_search_when_it_is_configured(self, mock_store):
        mock_store.hnsw_ef_search = 100

        set_ef_search, _, _, _ = mock_store._build_hnsw_queries()

        assert _render(set_ef_search) == "SET hnsw.ef_search = 100"

    def test_no_ef_search_statement_when_it_is_not_configured(self, mock_store):
        mock_store.hnsw_ef_search = None

        assert mock_store._build_hnsw_queries()[0] is None

    def test_the_drop_statement_names_the_index(self, mock_store):
        _, _, drop_index, _ = mock_store._build_hnsw_queries()

        assert _render(drop_index) == 'DROP INDEX IF EXISTS "public"."haystack_hnsw_index"'

    @pytest.mark.parametrize(
        ("vector_function", "operator_class"),
        [
            ("cosine_similarity", "vector_cosine_ops"),
            ("inner_product", "vector_ip_ops"),
            ("l2_distance", "vector_l2_ops"),
        ],
    )
    def test_the_index_uses_the_operator_class_of_the_vector_function(
        self, mock_store, vector_function, operator_class
    ):
        mock_store.vector_function = vector_function

        _, _, _, create_index = mock_store._build_hnsw_queries()

        assert operator_class in _render(create_index)

    def test_the_index_carries_the_configured_build_parameters(self, mock_store):
        mock_store.hnsw_index_creation_kwargs = {"m": 16, "ef_construction": 64}

        _, _, _, create_index = mock_store._build_hnsw_queries()

        rendered = _render(create_index)
        assert "m = 16" in rendered
        assert "ef_construction = 64" in rendered


class TestInsertStatement:
    def test_overwrite_updates_the_conflicting_row(self, mock_store):
        rendered = _render(mock_store._build_insert_statement(DuplicatePolicy.OVERWRITE))

        assert "ON CONFLICT" in rendered
        assert "DO UPDATE SET" in rendered
        assert rendered.endswith(" RETURNING id")


class TestCountQueries:
    def test_without_filters_it_counts_every_row(self, mock_store):
        params, query = mock_store._prepare_filters_count_documents(None)

        assert _render(query) == 'SELECT COUNT(*) FROM "public"."haystack"'
        assert params == ()

    def test_with_filters_it_appends_a_where_clause(self, mock_store):
        params, query = mock_store._prepare_filters_count_documents(EQUALS_BOOK)

        assert "WHERE" in _render(query)
        assert params == ("book",)


class TestCountUniqueMetadataQuery:
    def test_counts_the_distinct_values_of_each_field(self, mock_store):
        query, params = mock_store._build_count_unique_metadata_query(["kind", "year"], {})

        rendered = _render(query)
        assert "COUNT(DISTINCT meta->>'kind' ) AS \"kind\"" in rendered
        assert "COUNT(DISTINCT meta->>'year' ) AS \"year\"" in rendered
        assert params == ()

    def test_applies_the_filters(self, mock_store):
        query, params = mock_store._build_count_unique_metadata_query(["kind"], EQUALS_BOOK)

        assert "WHERE" in _render(query)
        assert params == ("book",)


class TestMetadataScanQuery:
    def test_skips_rows_without_metadata(self, mock_store):
        assert _render(mock_store._analyze_metadata_from_docs_query()) == (
            'SELECT meta FROM "public"."haystack" WHERE meta IS NOT NULL'
        )


class TestKeywordRetrievalQuery:
    def test_orders_by_score_and_limits(self, mock_store):
        query, params = mock_store._build_keyword_retrieval_query(query="a query", top_k=5)

        assert "ORDER BY score DESC LIMIT 5" in _render(query)
        assert params == ()

    def test_applies_the_filters(self, mock_store):
        query, params = mock_store._build_keyword_retrieval_query(query="a query", top_k=5, filters=EQUALS_BOOK)

        assert "AND" in _render(query)
        assert params == ("book",)


class TestMinMaxQuery:
    @pytest.mark.parametrize(("field_type", "cast"), [("integer", "::integer"), ("real", "::real")])
    def test_numeric_fields_are_cast_so_they_compare_numerically(self, mock_store, field_type, cast):
        rendered = _render(mock_store._build_min_max_query("year", field_type))

        assert f"(meta->>'year' ){cast}" in rendered
        assert "MIN(" in rendered
        assert "MAX(" in rendered

    def test_other_fields_compare_with_the_c_collation(self, mock_store):
        assert "meta->>'kind' COLLATE \"C\"" in _render(mock_store._build_min_max_query("kind", "text"))

    def test_rows_without_the_field_are_excluded(self, mock_store):
        assert "WHERE meta->>'year' IS NOT NULL" in _render(mock_store._build_min_max_query("year", "integer"))


class TestUniqueValuesQueries:
    def test_selects_distinct_values_using_the_jsonb_operator(self, mock_store):
        # `->` rather than `->>` keeps the original JSON type, so numbers sort numerically.
        count_query, select_query, params = mock_store._build_unique_values_queries("kind", None, 0, 10)

        assert "SELECT DISTINCT meta->'kind' AS value" in _render(select_query)
        assert "COUNT(DISTINCT meta->'kind' )" in _render(count_query)
        assert params == ()

    def test_paginates(self, mock_store):
        _, select_query, _ = mock_store._build_unique_values_queries("kind", None, 20, 5)

        assert "ORDER BY value LIMIT 5 OFFSET 20" in _render(select_query)

    def test_the_search_term_is_a_case_insensitive_substring_match(self, mock_store):
        _, select_query, params = mock_store._build_unique_values_queries("kind", "boo", 0, 10)

        assert "ILIKE" in _render(select_query)
        assert params == ("%boo%",)

    def test_applies_the_filters(self, mock_store):
        _, select_query, params = mock_store._build_unique_values_queries("kind", None, 0, 10, filters=EQUALS_BOOK)

        assert "AND" in _render(select_query)
        assert params == ("book",)

    @pytest.mark.parametrize(
        ("count_result", "records", "expected"),
        [
            ({"total": 2}, [{"value": "book"}, {"value": "paper"}], (["book", "paper"], 2)),
            ({"total": 1}, [{"value": "book"}, {"value": None}], (["book"], 1)),
            (None, [], ([], 0)),
        ],
        ids=["values", "null-dropped", "no-rows"],
    )
    def test_the_result_is_paired_with_its_total(self, count_result, records, expected):
        assert PgvectorDocumentStore._process_unique_values_result(count_result, records) == expected


class TestCursorBackedReads:
    def test_count_documents(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = (7,)

        assert mock_store_with_mock_connection.count_documents() == 7

    def test_count_documents_by_filter(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = (2,)

        assert mock_store_with_mock_connection.count_documents_by_filter(EQUALS_BOOK) == 2

    def test_filter_documents_converts_the_records(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [_record()]

        documents = mock_store_with_mock_connection.filter_documents(EQUALS_BOOK)

        assert [doc.id for doc in documents] == ["doc-1"]

    def test_get_metadata_fields_info(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [{"meta": {"year": 2024}}]

        assert mock_store_with_mock_connection.get_metadata_fields_info() == {"year": {"type": "integer"}}

    def test_get_metadata_field_min_max(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [{"meta": {"year": 2024}}]
        cursor.fetchone.return_value = {"min_value": 2020, "max_value": 2024}

        assert mock_store_with_mock_connection.get_metadata_field_min_max("meta.year") == {
            "min": 2020,
            "max": 2024,
        }

    def test_get_metadata_field_unique_values(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = {"total": 2}
        cursor.fetchall.return_value = [{"value": "book"}, {"value": "paper"}]

        assert mock_store_with_mock_connection.get_metadata_field_unique_values("meta.kind") == (
            ["book", "paper"],
            2,
        )

    def test_count_unique_metadata_by_filter(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = {"kind": 2}

        assert mock_store_with_mock_connection.count_unique_metadata_by_filter(EQUALS_BOOK, ["meta.kind"]) == {
            "kind": 2
        }

    def test_embedding_retrieval(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [_record()]

        documents = mock_store_with_mock_connection._embedding_retrieval([0.1] * 768, top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]

    def test_keyword_retrieval(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [_record()]

        documents = mock_store_with_mock_connection._keyword_retrieval("a query", top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]


class TestCursorBackedWrites:
    def test_write_documents_counts_the_ids_the_insert_returned(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.side_effect = [("doc-1",), ("doc-2",)]
        cursor.nextset.side_effect = [True, None]

        written = mock_store_with_mock_connection.write_documents(
            [Document(id="doc-1", content="alpha"), Document(id="doc-2", content="beta")],
            policy=DuplicatePolicy.OVERWRITE,
        )

        assert written == 2

    def test_a_duplicate_becomes_a_duplicate_document_error(self, mock_store_with_mock_connection, cursor):
        cursor.executemany.side_effect = IntegrityError("duplicate key")

        with pytest.raises(DuplicateDocumentError):
            mock_store_with_mock_connection.write_documents([Document(id="doc-1", content="alpha")])

        mock_store_with_mock_connection._connection.rollback.assert_called_once_with()

    def test_any_other_driver_error_becomes_a_document_store_error(self, mock_store_with_mock_connection, cursor):
        cursor.executemany.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="Could not write documents"):
            mock_store_with_mock_connection.write_documents([Document(id="doc-1", content="alpha")])

    def test_delete_documents(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_documents(["doc-1"])

        assert "DELETE" in _render(cursor.execute.call_args.args[0])

    def test_delete_documents_with_no_ids_is_a_no_op(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_documents([])

        cursor.execute.assert_not_called()

    def test_delete_all_documents(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_all_documents()

        assert "TRUNCATE TABLE" in _render(cursor.execute.call_args.args[0])

    def test_delete_by_filter(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_by_filter(EQUALS_BOOK)

        assert "DELETE" in _render(cursor.execute.call_args.args[0])

    def test_update_by_filter(self, mock_store_with_mock_connection, cursor):
        cursor.rowcount = 2

        mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {"seen": True})

        assert "UPDATE" in _render(cursor.execute.call_args.args[0])


class TestAsyncPaths:
    """The async twins, driven through a mocked async connection and cursor."""

    @pytest.mark.asyncio
    async def test_count_documents_async(self, async_store, async_cursor):
        async_cursor.fetchone = AsyncMock(return_value=(7,))

        assert await async_store.count_documents_async() == 7

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, async_store, async_cursor):
        async_cursor.fetchall = AsyncMock(return_value=[_record()])

        documents = await async_store.filter_documents_async(EQUALS_BOOK)

        assert [doc.id for doc in documents] == ["doc-1"]

    @pytest.mark.asyncio
    async def test_get_metadata_field_min_max_async(self, async_store, async_cursor):
        async_cursor.fetchall = AsyncMock(return_value=[{"meta": {"year": 2024}}])
        async_cursor.fetchone = AsyncMock(return_value={"min_value": 2020, "max_value": 2024})

        assert await async_store.get_metadata_field_min_max_async("meta.year") == {"min": 2020, "max": 2024}

    @pytest.mark.asyncio
    async def test_write_documents_async(self, async_store, async_cursor):
        async_cursor.__aiter__.return_value = iter([("doc-1",)])

        written = await async_store.write_documents_async(
            [Document(id="doc-1", content="alpha")], policy=DuplicatePolicy.OVERWRITE
        )

        assert written == 1

    @pytest.mark.asyncio
    async def test_write_documents_async_reports_a_duplicate(self, async_store, async_cursor):
        async_cursor.executemany = AsyncMock(side_effect=IntegrityError("duplicate key"))

        with pytest.raises(DuplicateDocumentError):
            await async_store.write_documents_async([Document(id="doc-1", content="alpha")])

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, async_store, async_cursor):
        await async_store.delete_documents_async(["doc-1"])

        assert "DELETE" in _render(async_cursor.execute.call_args.args[0])

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, async_store, async_cursor):
        async_cursor.fetchall = AsyncMock(return_value=[_record()])

        documents = await async_store._embedding_retrieval_async([0.1] * 768, top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]


@pytest.fixture
def store_with_real_setup_methods(monkeypatch):
    """
    A store with only the db-setup methods patched.

    The shared `mock_store` fixture also patches `delete_table` and `_handle_hnsw`,
    which are exactly the methods these tests exercise.
    """
    monkeypatch.setenv("PG_CONN_STR", "some-connection-string")

    with (
        patch.object(PgvectorDocumentStore, "_ensure_db_setup"),
        patch.object(PgvectorDocumentStore, "_ensure_db_setup_async", new=AsyncMock()),
        patch(f"{STORE_MODULE}.register_vector"),
    ):
        store = PgvectorDocumentStore(
            table_name="haystack",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            search_strategy="exact_nearest_neighbor",
        )
        cursor = _mock_cursor()
        cursor.execute.return_value = cursor
        store._connection = Mock()
        store._cursor = cursor
        store._dict_cursor = cursor

        async_cursor = _mock_cursor(spec=AsyncCursor)
        async_cursor.execute = AsyncMock(return_value=async_cursor)
        async_cursor.fetchone = AsyncMock(return_value=None)
        store._async_connection = Mock()
        store._async_cursor = async_cursor
        store._async_dict_cursor = async_cursor

        yield store


def _executed(cursor) -> list[str]:
    return [_render(call.args[0]) for call in cursor.execute.call_args_list]


class TestTableInitialization:
    def test_creates_the_table_and_the_keyword_index_when_they_are_missing(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        statements = _executed(store._cursor)
        assert any("CREATE TABLE" in sql for sql in statements)
        assert any("CREATE INDEX" in sql for sql in statements)
        assert store._table_initialized

    def test_leaves_an_existing_table_and_index_alone(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = (1,)

        store._initialize_table()

        statements = _executed(store._cursor)
        assert not any("CREATE TABLE" in sql for sql in statements)
        assert not any("CREATE INDEX" in sql for sql in statements)

    def test_drops_the_table_first_when_recreate_table_is_set(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.recreate_table = True
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        assert any("DROP TABLE IF EXISTS" in sql for sql in _executed(store._cursor))

    def test_builds_the_hnsw_index_when_that_search_strategy_is_selected(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.search_strategy = "hnsw"
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        assert any("USING hnsw" in sql for sql in _executed(store._cursor))

    def test_delete_table_drops_the_configured_table(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods

        store.delete_table()

        assert _executed(store._cursor) == ['DROP TABLE IF EXISTS "public"."haystack"']

    @pytest.mark.asyncio
    async def test_creates_the_table_asynchronously(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._async_cursor.fetchone = AsyncMock(return_value=None)

        await store._initialize_table_async()

        statements = _executed(store._async_cursor)
        assert any("CREATE TABLE" in sql for sql in statements)
        assert store._table_initialized


class TestHandleHnsw:
    def test_sets_ef_search_when_it_is_configured(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.hnsw_ef_search = 100
        store._cursor.fetchone.return_value = None

        store._handle_hnsw()

        assert any("SET hnsw.ef_search = 100" in sql for sql in _executed(store._cursor))

    def test_creates_the_index_when_it_is_missing(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = None

        store._handle_hnsw()

        statements = _executed(store._cursor)
        assert any("DROP INDEX IF EXISTS" in sql for sql in statements)
        assert any("USING hnsw" in sql for sql in statements)

    def test_keeps_an_existing_index_by_default(self, store_with_real_setup_methods, caplog):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = (1,)

        store._handle_hnsw()

        assert not any("USING hnsw" in sql for sql in _executed(store._cursor))
        assert "won't be recreated" in caplog.text

    def test_recreates_an_existing_index_on_request(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.hnsw_recreate_index_if_exists = True
        store._cursor.fetchone.return_value = (1,)

        store._handle_hnsw()

        assert any("USING hnsw" in sql for sql in _executed(store._cursor))

    @pytest.mark.asyncio
    async def test_creates_the_index_asynchronously_when_it_is_missing(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._async_cursor.fetchone = AsyncMock(return_value=None)

        await store._handle_hnsw_async()

        assert any("USING hnsw" in sql for sql in _executed(store._async_cursor))


class TestValidationAndErrorPaths:
    def test_halfvec_uses_its_own_operator_classes(self, mock_store):
        mock_store.vector_type = "halfvec"

        _, _, _, create_index = mock_store._build_hnsw_queries()

        assert "halfvec_cosine_ops" in _render(create_index)

    @pytest.mark.parametrize("vector_type", ["vector", "halfvec"])
    def test_rejects_a_vector_function_with_no_operator_class(self, mock_store, vector_type):
        mock_store.vector_type = vector_type
        mock_store.vector_function = "not-a-function"

        with pytest.raises(ValueError, match="Unsupported vector_function"):
            mock_store._build_hnsw_queries()

    def test_embedding_retrieval_rejects_an_empty_embedding(self, mock_store):
        with pytest.raises(ValueError, match="must be a non-empty list of floats"):
            mock_store._check_and_build_embedding_retrieval_query([], "cosine_similarity", 5)

    def test_embedding_retrieval_rejects_an_embedding_of_the_wrong_dimension(self, mock_store):
        with pytest.raises(ValueError, match="does not match PgvectorDocumentStore embedding dimension"):
            mock_store._check_and_build_embedding_retrieval_query([0.1, 0.2], "cosine_similarity", 5)

    def test_keyword_retrieval_requires_a_query(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            mock_store_with_mock_connection._keyword_retrieval("")

    def test_update_by_filter_requires_metadata(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {})

    def test_delete_by_filter_wraps_driver_errors(self, mock_store_with_mock_connection, cursor):
        cursor.execute.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="delete documents by filter"):
            mock_store_with_mock_connection.delete_by_filter(EQUALS_BOOK)

    def test_update_by_filter_wraps_driver_errors(self, mock_store_with_mock_connection, cursor):
        cursor.execute.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="update documents by filter"):
            mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {"seen": True})

    @pytest.mark.asyncio
    async def test_write_documents_async_wraps_driver_errors(self, async_store, async_cursor):
        async_cursor.executemany = AsyncMock(side_effect=Error("connection broken"))

        with pytest.raises(DocumentStoreError, match="Could not write documents"):
            await async_store.write_documents_async([Document(id="doc-1", content="alpha")])

    def test_count_documents_by_filter_without_a_row_reports_zero(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = None

        assert mock_store_with_mock_connection.count_documents_by_filter(EQUALS_BOOK) == 0

    def test_get_metadata_field_min_max_of_an_unknown_field(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [{"meta": {"year": 2024}}]

        assert mock_store_with_mock_connection.get_metadata_field_min_max("absent") == {"min": None, "max": None}


class TestAsyncSurface:
    """
    Every async method has its own body in this store rather than delegating to the
    sync one, so each needs to be executed. The behaviour they implement is asserted
    by the synchronous tests above; these check the await plumbing and the SQL verb.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args", "row", "rows", "expected"),
        [
            ("count_documents_by_filter_async", (EQUALS_BOOK,), (2,), [], 2),
            ("count_documents_by_filter_async", (EQUALS_BOOK,), None, [], 0),
            ("get_metadata_fields_info_async", (), None, [{"meta": {"year": 2024}}], {"year": {"type": "integer"}}),
            (
                "get_metadata_field_unique_values_async",
                ("meta.kind",),
                {"total": 1},
                [{"value": "book"}],
                (["book"], 1),
            ),
            ("count_unique_metadata_by_filter_async", (EQUALS_BOOK, ["meta.kind"]), {"kind": 2}, [], {"kind": 2}),
            (
                "get_metadata_field_min_max_async",
                ("absent",),
                None,
                [{"meta": {"year": 1}}],
                {"min": None, "max": None},
            ),
        ],
    )
    async def test_the_async_reads_return_what_the_cursor_gave_them(
        self, async_store, async_cursor, method, args, row, rows, expected
    ):
        async_cursor.fetchone = AsyncMock(return_value=row)
        async_cursor.fetchall = AsyncMock(return_value=rows)

        assert await getattr(async_store, method)(*args) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args", "verb"),
        [
            ("delete_documents_async", (["doc-1"],), "DELETE"),
            ("delete_all_documents_async", (), "TRUNCATE TABLE"),
            ("delete_by_filter_async", (EQUALS_BOOK,), "DELETE"),
            ("update_by_filter_async", (EQUALS_BOOK, {"seen": True}), "UPDATE"),
        ],
    )
    async def test_the_async_writes_issue_the_expected_statement(self, async_store, async_cursor, method, args, verb):
        async_cursor.rowcount = 1

        await getattr(async_store, method)(*args)

        assert verb in _render(async_cursor.execute.call_args.args[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args"),
        [("_embedding_retrieval_async", ([0.1] * 768,)), ("_keyword_retrieval_async", ("a query",))],
    )
    async def test_the_async_retrievals_convert_the_records(self, async_store, async_cursor, method, args):
        async_cursor.fetchall = AsyncMock(return_value=[_record()])

        documents = await getattr(async_store, method)(*args, top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]
