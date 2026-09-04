# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import psycopg
import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from psycopg import Connection, Cursor, Error, IntegrityError
from psycopg.adapt import AdaptersMap, Transformer
from psycopg.sql import SQL, Composed

from haystack_integrations.components.retrievers.alloydb import (
    AlloyDBEmbeddingRetriever,
    AlloyDBKeywordRetriever,
)
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore

EQUALS_BOOK = {"field": "meta.kind", "operator": "==", "value": "book"}

STORE_MODULE = "haystack_integrations.document_stores.alloydb.document_store"


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


@pytest.fixture
def cursor(mock_store_with_mock_connection) -> Mock:
    """A cursor whose results the store reads through, with no database behind it."""
    cursor = Mock(spec=Cursor)
    cursor.execute.return_value = cursor
    # `_execute_sql` renders the query against the cursor before running it, and psycopg
    # needs a real adapters map to do that.
    cursor.adapters = AdaptersMap(psycopg.adapters)
    cursor.connection = None
    # `write_documents` walks the RETURNING id result sets until nextset() runs out.
    cursor.fetchone.return_value = None
    cursor.nextset.return_value = None
    mock_store_with_mock_connection._cursor = cursor
    mock_store_with_mock_connection._dict_cursor = cursor
    mock_store_with_mock_connection._table_initialized = True
    return cursor


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

    def test_rejects_a_vector_function_with_no_operator_class(self, mock_store):
        mock_store.vector_function = "not-a-function"

        with pytest.raises(ValueError, match="Unsupported vector_function"):
            mock_store._build_hnsw_queries()


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

    def test_rejects_a_malformed_filter(self, mock_store):
        with pytest.raises(TypeError, match="Filters must be a dictionary"):
            mock_store._prepare_filters_count_documents("not a dict")


class TestCountUniqueMetadataQuery:
    def test_counts_the_distinct_values_of_each_field(self, mock_store):
        query, params = mock_store._build_count_unique_metadata_query(["kind", "year"], {})

        rendered = _render(query)
        assert "COUNT(DISTINCT meta->>'kind' ) AS \"kind\"" in rendered
        assert "COUNT(DISTINCT meta->>'year' ) AS \"year\"" in rendered
        assert params == ()

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            ({"kind": 3, "year": 5}, {"kind": 3, "year": 5}),
            ({"kind": 3}, {"kind": 3, "year": 0}),
            (None, {"kind": 0, "year": 0}),
        ],
        ids=["complete", "partial", "no-row"],
    )
    def test_the_result_is_reported_per_field(self, result, expected):
        assert AlloyDBDocumentStore._process_count_unique_metadata_result(result, ["kind", "year"]) == expected


class TestMetadataFieldTypes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "boolean"), (1, "integer"), (1.5, "real"), ("text", "text"), (None, "text"), ([1], "text")],
    )
    def test_infers_the_postgres_type_from_the_value(self, value, expected):
        assert AlloyDBDocumentStore._infer_metadata_field_type(value) == expected

    def test_analyzes_every_field_across_the_records(self):
        records = [{"meta": {"kind": "book", "year": 2024}}, {"meta": {"public": True}}]

        assert AlloyDBDocumentStore._analyze_metadata_fields_from_records(records) == {
            "kind": {"type": "text"},
            "year": {"type": "integer"},
            "public": {"type": "boolean"},
        }

    @pytest.mark.parametrize("meta", [None, "not a dict"], ids=["missing", "wrong-type"])
    def test_records_without_usable_metadata_are_skipped(self, meta):
        assert AlloyDBDocumentStore._analyze_metadata_fields_from_records([{"meta": meta}]) == {}

    def test_the_metadata_scan_query_skips_null_metadata(self, mock_store):
        rendered = _render(mock_store._analyze_metadata_from_docs_query())

        assert rendered == 'SELECT meta FROM "public"."haystack" WHERE meta IS NOT NULL'


class TestKeywordRetrievalQuery:
    def test_orders_by_score_and_limits(self, mock_store):
        query, params = mock_store._build_keyword_retrieval_query(top_k=5)

        rendered = _render(query)
        assert "ORDER BY score DESC LIMIT 5" in rendered
        assert params == ()


class TestEmbeddingRetrievalQuery:
    def test_builds_the_query_for_a_valid_embedding(self, mock_store):
        query, params = mock_store._check_and_build_embedding_retrieval_query(
            query_embedding=[0.1] * 768, vector_function="cosine_similarity", top_k=5
        )

        rendered = _render(query)
        assert "LIMIT 5" in rendered
        assert params == ()

    def test_rejects_an_empty_embedding(self, mock_store):
        with pytest.raises(ValueError, match="must be a non-empty list of floats"):
            mock_store._check_and_build_embedding_retrieval_query([], "cosine_similarity", 5)

    def test_rejects_an_embedding_of_the_wrong_dimension(self, mock_store):
        with pytest.raises(ValueError, match="does not match AlloyDBDocumentStore embedding dimension"):
            mock_store._check_and_build_embedding_retrieval_query([0.1, 0.2], "cosine_similarity", 5)

    def test_rejects_an_unknown_vector_function(self, mock_store):
        with pytest.raises(ValueError, match="vector_function must be one of"):
            mock_store._check_and_build_embedding_retrieval_query([0.1] * 768, "not-a-function", 5)


class TestMinMaxQuery:
    @pytest.mark.parametrize(("field_type", "cast"), [("integer", "::integer"), ("real", "::real")])
    def test_numeric_fields_are_cast_so_they_compare_numerically(self, mock_store, field_type, cast):
        rendered = _render(mock_store._build_min_max_query("year", field_type))

        assert f"(meta->>'year'){cast}" in rendered
        assert "MIN(" in rendered
        assert "MAX(" in rendered

    def test_other_fields_compare_with_the_c_collation(self, mock_store):
        rendered = _render(mock_store._build_min_max_query("kind", "text"))

        assert "meta->>'kind' COLLATE \"C\"" in rendered

    def test_rows_without_the_field_are_excluded(self, mock_store):
        assert "WHERE meta->>'year' IS NOT NULL" in _render(mock_store._build_min_max_query("year", "integer"))


class TestUniqueValuesQueries:
    def test_applies_the_filters_and_the_search_term_together(self, mock_store):
        _, select_query, params = mock_store._build_unique_values_queries("kind", EQUALS_BOOK, "boo", 0, 10)

        assert "AND" in _render(select_query)
        assert params == ("book", "%boo%")

    def test_selects_distinct_values_using_the_jsonb_operator(self, mock_store):
        # `->` rather than `->>` keeps the original JSON type, so numbers sort numerically.
        count_query, select_query, params = mock_store._build_unique_values_queries("kind", None, None, 0, 10)

        assert "SELECT DISTINCT meta->'kind' AS value" in _render(select_query)
        assert "COUNT(DISTINCT meta->'kind' )" in _render(count_query)
        assert params == ()

    def test_paginates(self, mock_store):
        _, select_query, _ = mock_store._build_unique_values_queries("kind", None, None, 20, 5)

        assert "ORDER BY value LIMIT 5 OFFSET 20" in _render(select_query)

    def test_the_search_term_is_a_case_insensitive_substring_match(self, mock_store):
        _, select_query, params = mock_store._build_unique_values_queries("kind", None, "boo", 0, 10)

        assert "ILIKE" in _render(select_query)
        assert params == ("%boo%",)

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
        assert AlloyDBDocumentStore._process_unique_values_result(count_result, records) == expected


class TestCursorBackedReads:
    def test_count_documents(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = (7,)

        assert mock_store_with_mock_connection.count_documents() == 7

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

        counts = mock_store_with_mock_connection.count_unique_metadata_by_filter(EQUALS_BOOK, ["meta.kind"])

        assert counts == {"kind": 2}

    def test_count_unique_metadata_by_filter_rejects_no_fields(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="metadata_fields"):
            mock_store_with_mock_connection.count_unique_metadata_by_filter(EQUALS_BOOK, [])

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
        # The store walks the RETURNING id result sets until nextset() runs out.
        cursor.fetchone.side_effect = [("doc-1",), ("doc-2",)]
        cursor.nextset.side_effect = [True, None]

        written = mock_store_with_mock_connection.write_documents(
            [Document(id="doc-1", content="alpha"), Document(id="doc-2", content="beta")],
            policy=DuplicatePolicy.OVERWRITE,
        )

        assert written == 2

    def test_write_documents_rejects_anything_that_is_not_a_document(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            mock_store_with_mock_connection.write_documents(["not a document"])

    def test_delete_documents(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_documents(["doc-1"])

        assert "DELETE" in _render(cursor.execute.call_args.args[0])

    def test_delete_documents_with_no_ids_is_a_no_op(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_documents([])

        cursor.execute.assert_not_called()

    def test_delete_all_documents_truncates_the_table(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_all_documents()

        assert _render(cursor.execute.call_args.args[0]) == 'TRUNCATE TABLE "public"."haystack"'

    def test_delete_by_filter(self, mock_store_with_mock_connection, cursor):
        mock_store_with_mock_connection.delete_by_filter(EQUALS_BOOK)

        assert "DELETE" in _render(cursor.execute.call_args.args[0])

    def test_update_by_filter(self, mock_store_with_mock_connection, cursor):
        cursor.rowcount = 2

        mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {"seen": True})

        assert "UPDATE" in _render(cursor.execute.call_args.args[0])


class TestRetrievers:
    def test_the_embedding_retriever_delegates_to_the_store(self):
        store = Mock(spec=AlloyDBDocumentStore)
        store._embedding_retrieval.return_value = [Document(content="alpha")]
        retriever = AlloyDBEmbeddingRetriever(document_store=store, top_k=3)

        documents = retriever.run(query_embedding=[0.1, 0.2])["documents"]

        assert [doc.content for doc in documents] == ["alpha"]
        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2], filters={}, top_k=3, vector_function=None
        )

    def test_the_keyword_retriever_delegates_to_the_store(self):
        store = Mock(spec=AlloyDBDocumentStore)
        store._keyword_retrieval.return_value = [Document(content="alpha")]
        retriever = AlloyDBKeywordRetriever(document_store=store, top_k=3)

        documents = retriever.run(query="a query")["documents"]

        assert [doc.content for doc in documents] == ["alpha"]
        store._keyword_retrieval.assert_called_once_with(query="a query", filters={}, top_k=3)

    @pytest.mark.usefixtures("patches_for_unit_tests")
    def test_the_keyword_retriever_round_trips_through_a_dictionary(self, mock_store, monkeypatch):
        monkeypatch.setenv(
            "ALLOYDB_INSTANCE_URI",
            "projects/test-project/locations/us-central1/clusters/test-cluster/instances/test-instance",
        )
        monkeypatch.setenv("ALLOYDB_USER", "postgres")
        monkeypatch.setenv("ALLOYDB_PASSWORD", "postgres")
        retriever = AlloyDBKeywordRetriever(document_store=mock_store, filters=EQUALS_BOOK, top_k=7)

        deserialized = AlloyDBKeywordRetriever.from_dict(retriever.to_dict())

        assert deserialized.filters == EQUALS_BOOK
        assert deserialized.top_k == 7
        assert isinstance(deserialized.document_store, AlloyDBDocumentStore)


@pytest.fixture
def store_with_real_setup_methods(monkeypatch):
    """
    A store with only `_ensure_db_setup` patched.

    The shared `mock_store` fixture also patches `_initialize_table`, `delete_table`
    and `_handle_hnsw`, which are exactly the methods these tests exercise.
    """
    monkeypatch.setenv(
        "ALLOYDB_INSTANCE_URI",
        "projects/test-project/locations/us-central1/clusters/test-cluster/instances/test-instance",
    )
    monkeypatch.setenv("ALLOYDB_USER", "postgres")
    monkeypatch.setenv("ALLOYDB_PASSWORD", "postgres")

    with patch(f"{STORE_MODULE}.AlloyDBDocumentStore._ensure_db_setup"):
        store = AlloyDBDocumentStore(
            table_name="haystack",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            search_strategy="exact_nearest_neighbor",
        )
        cursor = Mock(spec=Cursor)
        cursor.execute.return_value = cursor
        cursor.adapters = AdaptersMap(psycopg.adapters)
        cursor.connection = None
        store._connection = Mock(spec=Connection)
        store._cursor = cursor
        store._dict_cursor = cursor
        yield store


def _executed(store) -> list[str]:
    return [_render(call.args[0]) for call in store._cursor.execute.call_args_list]


class TestTableInitialization:
    def test_creates_the_table_and_the_keyword_index_when_they_are_missing(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        statements = _executed(store)
        assert any("CREATE TABLE" in sql for sql in statements)
        assert any("CREATE INDEX" in sql for sql in statements)
        assert store._table_initialized

    def test_drops_the_table_first_when_recreate_table_is_set(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.recreate_table = True
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        assert any("DROP TABLE IF EXISTS" in sql for sql in _executed(store))

    def test_builds_the_hnsw_index_when_that_search_strategy_is_selected(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.search_strategy = "hnsw"
        store._cursor.fetchone.return_value = None

        store._initialize_table()

        assert any("USING hnsw" in sql for sql in _executed(store))

    def test_delete_table_drops_the_configured_table(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods

        store.delete_table()

        assert _executed(store) == ['DROP TABLE IF EXISTS "public"."haystack"']


class TestHandleHnsw:
    def test_sets_ef_search_when_it_is_configured(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store.hnsw_ef_search = 100
        store._cursor.fetchone.return_value = None

        store._handle_hnsw()

        assert any("SET hnsw.ef_search = 100" in sql for sql in _executed(store))

    def test_creates_the_index_when_it_is_missing(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = None

        store._handle_hnsw()

        statements = _executed(store)
        assert any("DROP INDEX IF EXISTS" in sql for sql in statements)
        assert any("USING hnsw" in sql for sql in statements)

    def test_keeps_an_existing_index_by_default(self, store_with_real_setup_methods, caplog):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = (1,)

        store._handle_hnsw()

        assert not any("USING hnsw" in sql for sql in _executed(store))
        assert "won't be recreated" in caplog.text


class TestWriteErrors:
    def test_a_duplicate_becomes_a_duplicate_document_error(self, mock_store_with_mock_connection, cursor):
        cursor.executemany.side_effect = IntegrityError("duplicate key")

        with pytest.raises(DuplicateDocumentError):
            mock_store_with_mock_connection.write_documents([Document(id="doc-1", content="alpha")])

        mock_store_with_mock_connection._connection.rollback.assert_called_once_with()

    def test_any_other_driver_error_becomes_a_document_store_error(self, mock_store_with_mock_connection, cursor):
        cursor.executemany.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="Could not write documents"):
            mock_store_with_mock_connection.write_documents([Document(id="doc-1", content="alpha")])


class TestFilterBasedErrorPaths:
    def test_count_documents_without_a_row_reports_zero(self, mock_store_with_mock_connection, cursor):
        cursor.fetchone.return_value = None

        assert mock_store_with_mock_connection.count_documents() == 0

    def test_get_metadata_field_min_max_without_a_row(self, mock_store_with_mock_connection, cursor):
        cursor.fetchall.return_value = [{"meta": {"year": 2024}}]
        cursor.fetchone.return_value = None

        assert mock_store_with_mock_connection.get_metadata_field_min_max("year") == {"min": None, "max": None}

    def test_delete_by_filter_wraps_driver_errors(self, mock_store_with_mock_connection, cursor):
        cursor.execute.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="Could not delete documents by filter"):
            mock_store_with_mock_connection.delete_by_filter(EQUALS_BOOK)

    def test_update_by_filter_wraps_driver_errors(self, mock_store_with_mock_connection, cursor):
        cursor.execute.side_effect = Error("connection broken")

        with pytest.raises(DocumentStoreError, match="Could not update documents by filter"):
            mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {"seen": True})

    def test_update_by_filter_requires_metadata(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="meta must be a non-empty dictionary"):
            mock_store_with_mock_connection.update_by_filter(EQUALS_BOOK, {})

    def test_keyword_retrieval_requires_a_query(self, mock_store_with_mock_connection):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            mock_store_with_mock_connection._keyword_retrieval("")

    @pytest.mark.parametrize(
        ("vector_function", "operator"),
        [("cosine_similarity", "<=>"), ("inner_product", "<#>"), ("l2_distance", "<->")],
    )
    def test_each_vector_function_scores_with_its_own_operator(self, mock_store, vector_function, operator):
        query, _ = mock_store._check_and_build_embedding_retrieval_query([0.1] * 768, vector_function, 5)

        assert operator in _render(query)

    def test_hnsw_index_creation_carries_the_configured_build_parameters(self, mock_store):
        mock_store.hnsw_index_creation_kwargs = {"m": 16, "ef_construction": 64}

        _, _, _, create_index = mock_store._build_hnsw_queries()

        rendered = _render(create_index)
        assert "m = 16" in rendered
        assert "ef_construction = 64" in rendered

    def test_close_releases_the_connector(self, mock_store_with_mock_connection):
        mock_store_with_mock_connection._connector = Mock()

        mock_store_with_mock_connection.close()

        assert mock_store_with_mock_connection._connector is None
        assert not mock_store_with_mock_connection._table_initialized


class TestBranchesReachedThroughConfiguration:
    """Alternative configurations of the same builders, kept compact."""

    @pytest.mark.parametrize(
        ("attribute", "value", "expected"),
        [
            ("hnsw_recreate_index_if_exists", True, True),
            ("hnsw_recreate_index_if_exists", False, False),
        ],
    )
    def test_an_existing_hnsw_index_is_only_recreated_on_request(
        self, store_with_real_setup_methods, attribute, value, expected
    ):
        store = store_with_real_setup_methods
        setattr(store, attribute, value)
        store._cursor.fetchone.return_value = (1,)

        store._handle_hnsw()

        assert any("USING hnsw" in sql for sql in _executed(store)) is expected

    def test_an_existing_table_and_index_are_left_alone(self, store_with_real_setup_methods):
        store = store_with_real_setup_methods
        store._cursor.fetchone.return_value = (1,)

        store._initialize_table()

        assert not any("CREATE TABLE" in sql for sql in _executed(store))

    @pytest.mark.parametrize(
        ("method", "args", "row", "rows", "expected"),
        [
            ("count_documents_by_filter", (EQUALS_BOOK,), (2,), [], 2),
            ("count_documents_by_filter", (EQUALS_BOOK,), None, [], 0),
            ("get_metadata_field_min_max", ("absent",), None, [{"meta": {"year": 1}}], {"min": None, "max": None}),
        ],
    )
    def test_reads_return_what_the_cursor_gave_them(
        self, mock_store_with_mock_connection, cursor, method, args, row, rows, expected
    ):
        cursor.fetchone.return_value = row
        cursor.fetchall.return_value = rows

        assert getattr(mock_store_with_mock_connection, method)(*args) == expected

    def test_the_retrievers_pass_their_runtime_overrides_to_the_store(self):
        store = Mock(spec=AlloyDBDocumentStore)
        store._embedding_retrieval.return_value = []
        retriever = AlloyDBEmbeddingRetriever(document_store=store, top_k=3)

        retriever.run(query_embedding=[0.1], filters=EQUALS_BOOK, top_k=7)

        kwargs = store._embedding_retrieval.call_args.kwargs
        assert (kwargs["top_k"], kwargs["filters"]) == (7, EQUALS_BOOK)
