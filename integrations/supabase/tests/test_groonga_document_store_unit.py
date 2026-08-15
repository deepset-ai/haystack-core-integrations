# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore

STORE_MODULE = "haystack_integrations.document_stores.supabase.groonga_document_store"


class _NotProxy:
    """Stands in for PostgREST's `query.not_` accessor."""

    def __init__(self, query: "RecordingQuery") -> None:
        self._query = query

    def is_(self, column: str, value: Any) -> "RecordingQuery":
        return self._query._record("not.is_", column, value)

    def in_(self, column: str, value: Any) -> "RecordingQuery":
        return self._query._record("not.in_", column, value)


class RecordingQuery:
    """
    A stand-in for the PostgREST query builder that records what the filter
    translation asks it to do, so tests can assert on the generated query.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.not_ = _NotProxy(self)

    def _record(self, name: str, *args: Any) -> "RecordingQuery":
        self.calls.append((name, *args))
        return self

    def eq(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("eq", column, value)

    def neq(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("neq", column, value)

    def gt(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("gt", column, value)

    def gte(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("gte", column, value)

    def lt(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("lt", column, value)

    def lte(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("lte", column, value)

    def in_(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("in_", column, value)

    def is_(self, column: str, value: Any) -> "RecordingQuery":
        return self._record("is_", column, value)

    def or_(self, expression: str) -> "RecordingQuery":
        return self._record("or_", expression)


@pytest.fixture
def query() -> RecordingQuery:
    return RecordingQuery()


def _apply(query: RecordingQuery, filters: dict) -> list[tuple]:
    SupabaseGroongaDocumentStore._apply_filters(query, filters)
    return query.calls


class TestMetaCol:
    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            # Numeric values use the JSONB accessor so PostgREST compares numerically.
            ("meta.year", 2024, "meta->year"),
            ("meta.rating", 4.5, "meta->rating"),
            # Text, booleans and None use the text accessor.
            ("meta.kind", "book", "meta->>kind"),
            ("meta.public", True, "meta->>public"),
            ("meta.missing", None, "meta->>missing"),
            # A list of numbers is numeric; anything else is not.
            ("meta.years", [2020, 2024], "meta->years"),
            ("meta.mixed", [2020, "a"], "meta->>mixed"),
            ("meta.flags", [True, False], "meta->>flags"),
            ("meta.empty", [], "meta->>empty"),
            # Top-level columns are passed through untouched.
            ("content", "text", "content"),
            ("id", "abc", "id"),
        ],
    )
    def test_chooses_the_column_expression_from_the_value_type(self, field, value, expected):
        assert SupabaseGroongaDocumentStore._meta_col(field, value) == expected


class TestNormalizeValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "true"), (False, "false"), (2024, 2024), ("book", "book"), (None, None)],
    )
    def test_booleans_become_jsonb_text_literals(self, value, expected):
        assert SupabaseGroongaDocumentStore._normalize_value(value) == expected


class TestApplyCondition:
    def test_equality(self, query):
        assert _apply(query, {"field": "meta.kind", "operator": "==", "value": "book"}) == [
            ("eq", "meta->>kind", "book")
        ]

    def test_equality_against_none_becomes_an_is_null_check(self, query):
        assert _apply(query, {"field": "meta.kind", "operator": "==", "value": None}) == [
            ("is_", "meta->>kind", "null")
        ]

    def test_inequality_against_none_becomes_a_not_is_null_check(self, query):
        assert _apply(query, {"field": "meta.kind", "operator": "!=", "value": None}) == [
            ("not.is_", "meta->>kind", "null")
        ]

    def test_inequality_on_a_meta_field_also_matches_documents_missing_the_field(self, query):
        # SQL: NULL != value is NULL, not TRUE, so absent fields need an explicit clause.
        assert _apply(query, {"field": "meta.kind", "operator": "!=", "value": "book"}) == [
            ("or_", "meta->>kind.neq.book,meta->>kind.is.null")
        ]

    @pytest.mark.parametrize(
        ("operator", "expected_call"),
        [(">", "gt"), (">=", "gte"), ("<", "lt"), ("<=", "lte")],
    )
    def test_ordering_operators(self, query, operator, expected_call):
        assert _apply(query, {"field": "meta.year", "operator": operator, "value": 2024}) == [
            (expected_call, "meta->year", 2024)
        ]

    def test_ordering_operators_against_none_match_nothing(self, query):
        # An impossible equality is the store's way of returning an empty result.
        assert _apply(query, {"field": "meta.year", "operator": ">", "value": None}) == [("eq", "id", "")]

    def test_ordering_operators_reject_lists(self, query):
        with pytest.raises(FilterError, match="does not support list values"):
            _apply(query, {"field": "meta.year", "operator": ">", "value": [1, 2]})

    def test_ordering_operators_reject_non_date_strings(self, query):
        with pytest.raises(FilterError, match="does not support string values"):
            _apply(query, {"field": "meta.kind", "operator": ">", "value": "book"})

    def test_in(self, query):
        assert _apply(query, {"field": "meta.year", "operator": "in", "value": [2020, 2024]}) == [
            ("in_", "meta->year", [2020, 2024])
        ]

    def test_in_requires_a_list(self, query):
        with pytest.raises(FilterError, match="'in' requires a list value"):
            _apply(query, {"field": "meta.year", "operator": "in", "value": 2020})

    def test_not_in_on_a_meta_field_also_matches_documents_missing_the_field(self, query):
        assert _apply(query, {"field": "meta.year", "operator": "not in", "value": [2020, None]}) == [
            ("or_", "meta->year.not.in.(2020),meta->>year.is.null")
        ]

    def test_not_in_requires_a_list(self, query):
        with pytest.raises(FilterError, match="'not in' requires a list value"):
            _apply(query, {"field": "meta.year", "operator": "not in", "value": 2020})

    def test_an_unknown_operator_leaves_the_query_untouched(self, query):
        assert _apply(query, {"field": "meta.year", "operator": "~=", "value": 2020}) == []

    def test_requires_an_operator(self, query):
        with pytest.raises(FilterError, match="must include an 'operator' key"):
            _apply(query, {"field": "meta.year", "value": 2020})

    def test_requires_a_value(self, query):
        with pytest.raises(FilterError, match="must include a 'value' key"):
            _apply(query, {"field": "meta.year", "operator": "=="})


class TestApplyFilters:
    def test_empty_filters_leave_the_query_untouched(self, query):
        assert _apply(query, {}) == []

    def test_and_applies_every_condition_in_turn(self, query):
        calls = _apply(
            query,
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.kind", "operator": "==", "value": "book"},
                    {"field": "meta.year", "operator": ">", "value": 2020},
                ],
            },
        )

        assert calls == [("eq", "meta->>kind", "book"), ("gt", "meta->year", 2020)]

    def test_or_becomes_a_single_postgrest_or_expression(self, query):
        calls = _apply(
            query,
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.kind", "operator": "==", "value": "book"},
                    {"field": "meta.year", "operator": ">=", "value": 2020},
                ],
            },
        )

        # OR expressions always use the text accessor: PostgREST cannot parse -> inside them.
        assert calls == [("or_", "meta->>kind.eq.book,meta->>year.gte.2020")]

    def test_or_rejects_nested_logical_operators(self, query):
        with pytest.raises(FilterError, match="Nested logical operators inside OR"):
            _apply(query, {"operator": "OR", "conditions": [{"operator": "AND", "conditions": []}]})

    def test_or_rejects_operators_it_cannot_express(self, query):
        with pytest.raises(FilterError, match="inside OR filter is not supported"):
            _apply(
                query,
                {"operator": "OR", "conditions": [{"field": "meta.year", "operator": "in", "value": [1]}]},
            )

    def test_not_negates_each_condition(self, query):
        calls = _apply(
            query,
            {"operator": "NOT", "conditions": [{"field": "meta.year", "operator": ">", "value": 2020}]},
        )

        assert calls == [("or_", "meta->>year.lte.2020")]

    def test_not_equals_also_matches_documents_missing_the_field(self, query):
        calls = _apply(
            query,
            {"operator": "NOT", "conditions": [{"field": "meta.kind", "operator": "==", "value": "book"}]},
        )

        assert calls == [("or_", "meta->>kind.neq.book,meta->>kind.is.null")]

    def test_not_rejects_nested_logical_operators(self, query):
        with pytest.raises(FilterError, match="Nested logical operators inside NOT"):
            _apply(query, {"operator": "NOT", "conditions": [{"operator": "AND", "conditions": []}]})

    def test_not_rejects_operators_it_cannot_express(self, query):
        with pytest.raises(FilterError, match="inside NOT filter is not supported"):
            _apply(
                query,
                {"operator": "NOT", "conditions": [{"field": "meta.year", "operator": "in", "value": [1]}]},
            )

    def test_requires_a_logical_operator(self, query):
        with pytest.raises(FilterError, match="must include an 'operator' key"):
            _apply(query, {"conditions": []})

    def test_requires_conditions(self, query):
        with pytest.raises(FilterError, match="must include a 'conditions' key"):
            _apply(query, {"operator": "AND"})

    def test_rejects_an_unknown_logical_operator(self, query):
        with pytest.raises(FilterError, match=r"is not supported\. Supported logical operators"):
            _apply(query, {"operator": "XOR", "conditions": []})


class TestMatchCondition:
    """In-memory matching, used to filter PGroonga results after retrieval."""

    DOC = Document(id="a", content="alpha", meta={"year": 2024, "kind": "book"})

    @pytest.mark.parametrize(
        ("condition", "expected"),
        [
            ({"field": "meta.year", "operator": "==", "value": 2024}, True),
            ({"field": "meta.year", "operator": "!=", "value": 2020}, True),
            ({"field": "meta.year", "operator": ">", "value": 2020}, True),
            ({"field": "meta.year", "operator": ">=", "value": 2024}, True),
            ({"field": "meta.year", "operator": "<", "value": 2020}, False),
            ({"field": "meta.year", "operator": "<=", "value": 2024}, True),
            ({"field": "meta.year", "operator": "in", "value": [2024]}, True),
            ({"field": "meta.year", "operator": "not in", "value": [2024]}, False),
            ({"field": "content", "operator": "==", "value": "alpha"}, True),
            ({"field": "meta.missing", "operator": "==", "value": None}, True),
            # Ordering comparisons never match a document that lacks the field.
            ({"field": "meta.missing", "operator": ">", "value": 1}, False),
            ({"field": "meta.year", "operator": "~=", "value": 1}, True),
        ],
    )
    def test_operators(self, condition, expected):
        assert SupabaseGroongaDocumentStore._match_condition(self.DOC, condition) is expected


class TestSetupTable:
    @pytest.mark.usefixtures("groonga_store")
    def test_creates_the_table_and_the_pgroonga_index(self, mock_supabase_client):
        statements = [call.args[1]["query"] for call in mock_supabase_client.rpc.call_args_list]

        assert any("CREATE TABLE IF NOT EXISTS test_groonga_documents" in sql for sql in statements)
        assert any("USING pgroonga (content)" in sql for sql in statements)
        assert not any("DROP TABLE" in sql for sql in statements)

    def test_drops_the_table_first_when_recreate_table_is_set(self, mock_supabase_client, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake.supabase.co", table_name="docs", recreate_table=True
        )

        store.warm_up()

        statements = [call.args[1]["query"] for call in mock_supabase_client.rpc.call_args_list]
        assert any("DROP TABLE IF EXISTS docs" in sql for sql in statements)

    @pytest.mark.parametrize(
        "method",
        ["count_documents", "filter_documents", "delete_all_documents", "_setup_table"],
    )
    def test_using_the_store_before_warm_up_raises(self, method, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake.supabase.co")

        with pytest.raises(RuntimeError, match="Call warm_up"):
            getattr(store, method)()


class TestCountDocuments:
    def test_returns_the_reported_count(self, groonga_store, mock_supabase_client):
        mock_supabase_client.table.return_value.execute.return_value = MagicMock(count=7)

        assert groonga_store.count_documents() == 7


class TestWriteDocuments:
    def test_overwrite_upserts_each_document(self, groonga_store, mock_supabase_client):
        table = mock_supabase_client.table.return_value

        written = groonga_store.write_documents(
            [Document(id="a", content="alpha", meta={"kind": "book"})], policy=DuplicatePolicy.OVERWRITE
        )

        assert written == 1
        table.upsert.assert_called_once_with({"id": "a", "content": "alpha", "meta": {"kind": "book"}, "score": None})

    def test_skip_leaves_an_existing_document_alone(self, groonga_store, mock_supabase_client):
        table = mock_supabase_client.table.return_value
        table.execute.return_value = MagicMock(data=[{"id": "a"}])

        written = groonga_store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.SKIP)

        assert written == 0
        table.insert.assert_not_called()

    def test_fail_raises_on_an_existing_document(self, groonga_store, mock_supabase_client):
        mock_supabase_client.table.return_value.execute.return_value = MagicMock(data=[{"id": "a"}])

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            groonga_store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.FAIL)

    def test_writing_nothing_is_a_no_op(self, groonga_store):
        assert groonga_store.write_documents([]) == 0

    def test_rejects_something_that_is_not_a_list(self, groonga_store):
        with pytest.raises(ValueError, match="expects a list of Document objects"):
            groonga_store.write_documents("not a list")

    def test_rejects_a_list_holding_something_other_than_documents(self, groonga_store):
        with pytest.raises(ValueError, match="expects Document objects"):
            groonga_store.write_documents([Document(id="a"), "not a document"])


class TestDeleteAndUpdate:
    def test_delete_documents_deletes_the_given_ids(self, groonga_store, mock_supabase_client):
        table = mock_supabase_client.table.return_value

        groonga_store.delete_documents(["a", "b"])

        table.delete.assert_called_once()
        table.in_.assert_called_once_with("id", ["a", "b"])

    def test_delete_all_documents_matches_every_row(self, groonga_store, mock_supabase_client):
        table = mock_supabase_client.table.return_value

        groonga_store.delete_all_documents()

        table.delete.assert_called_once()
        table.neq.assert_called_once_with("id", "")

    def test_delete_by_filter_returns_the_number_deleted(self, groonga_store, mock_supabase_client):
        mock_supabase_client.table.return_value.execute.return_value = MagicMock(
            data=[{"id": "a", "meta": {}}, {"id": "b", "meta": {}}]
        )

        assert groonga_store.delete_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}) == 2

    def test_update_by_filter_merges_the_new_metadata(self, groonga_store, mock_supabase_client):
        table = mock_supabase_client.table.return_value
        table.execute.return_value = MagicMock(data=[{"id": "a", "content": "alpha", "meta": {"kind": "book"}}])

        updated = groonga_store.update_by_filter(
            {"field": "meta.kind", "operator": "==", "value": "book"}, {"seen": True}
        )

        assert updated == 1
        assert table.upsert.call_args.args[0]["meta"] == {"kind": "book", "seen": True}

    def test_update_by_filter_requires_warm_up(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake.supabase.co")

        with pytest.raises(RuntimeError, match="Call warm_up"):
            store.update_by_filter({}, {})


class TestGroongaRetrieval:
    def test_calls_the_search_function_with_the_query_and_table(self, groonga_store, mock_supabase_client):
        groonga_store._groonga_retrieval(query="search text", top_k=5)

        mock_supabase_client.rpc.assert_called_with(
            "groonga_search", {"query_text": "search text", "table_name": "test_groonga_documents", "top_k": 5}
        )

    def test_converts_the_returned_rows(self, groonga_store, mock_supabase_client):
        mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"id": "a", "content": "alpha", "meta": {}, "score": 1.0}]
        )

        docs = groonga_store._groonga_retrieval(query="alpha")

        assert [doc.id for doc in docs] == ["a"]

    def test_applies_filters_after_retrieval(self, groonga_store, mock_supabase_client):
        mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {"id": "a", "content": "alpha", "meta": {"kind": "book"}},
                {"id": "b", "content": "beta", "meta": {"kind": "paper"}},
            ]
        )

        docs = groonga_store._groonga_retrieval(
            query="alpha", filters={"field": "meta.kind", "operator": "==", "value": "book"}
        )

        assert [doc.id for doc in docs] == ["a"]


@pytest.fixture
def async_client():
    """An async Supabase client: the builder calls stay sync, only execute is awaited."""
    with patch(f"{STORE_MODULE}.acreate_client", new_callable=AsyncMock) as acreate:
        client = MagicMock()
        acreate.return_value = client

        table = MagicMock()
        client.table.return_value = table
        for method in ("select", "insert", "upsert", "delete", "eq", "neq", "in_"):
            getattr(table, method).return_value = table
        table.execute = AsyncMock(return_value=MagicMock(data=[], count=0))
        client.rpc.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))

        yield client


@pytest.fixture
async def async_store(async_client, monkeypatch) -> SupabaseGroongaDocumentStore:  # noqa: ARG001
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
    return SupabaseGroongaDocumentStore(supabase_url="https://fake.supabase.co", table_name="docs")


class TestAsyncPaths:
    async def test_count_documents_async(self, async_store, async_client):
        async_client.table.return_value.execute = AsyncMock(return_value=MagicMock(count=3))

        assert await async_store.count_documents_async() == 3

    async def test_write_documents_async_overwrite(self, async_store, async_client):
        written = await async_store.write_documents_async(
            [Document(id="a", content="alpha")], policy=DuplicatePolicy.OVERWRITE
        )

        assert written == 1
        async_client.table.return_value.upsert.assert_called_once()

    async def test_write_documents_async_fail_raises_on_a_duplicate(self, async_store, async_client):
        async_client.table.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "a"}]))

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            await async_store.write_documents_async([Document(id="a")], policy=DuplicatePolicy.FAIL)

    async def test_delete_documents_async(self, async_store, async_client):
        table = async_client.table.return_value

        await async_store.delete_documents_async(["a"])

        table.in_.assert_called_once_with("id", ["a"])

    async def test_delete_by_filter_async(self, async_store, async_client):
        async_client.table.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "a", "meta": {}}]))

        assert await async_store.delete_by_filter_async({"field": "id", "operator": "==", "value": "a"}) == 1

    async def test_update_by_filter_async_merges_the_new_metadata(self, async_store, async_client):
        table = async_client.table.return_value
        table.execute = AsyncMock(
            return_value=MagicMock(data=[{"id": "a", "content": "alpha", "meta": {"kind": "book"}}])
        )

        updated = await async_store.update_by_filter_async({"field": "id", "operator": "==", "value": "a"}, {"n": 1})

        assert updated == 1
        assert table.upsert.call_args.args[0]["meta"] == {"kind": "book", "n": 1}

    async def test_groonga_retrieval_async_applies_filters_after_retrieval(self, async_store, async_client):
        async_client.rpc.return_value.execute = AsyncMock(
            return_value=MagicMock(
                data=[
                    {"id": "a", "content": "alpha", "meta": {"kind": "book"}},
                    {"id": "b", "content": "beta", "meta": {"kind": "paper"}},
                ]
            )
        )

        docs = await async_store._groonga_retrieval_async(
            query="alpha", filters={"field": "meta.kind", "operator": "==", "value": "book"}
        )

        assert [doc.id for doc in docs] == ["a"]

    async def test_setup_table_async_requires_a_client(self, async_store):
        with pytest.raises(RuntimeError, match="Async client not initialized"):
            await async_store._setup_table_async()
