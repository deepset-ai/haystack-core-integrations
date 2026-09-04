# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


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


class TestInit:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store.table_name == "haystack_groonga_documents"
        assert store.recreate_table is False
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None

    def test_init_custom_params(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_custom_table",
            recreate_table=True,
        )
        assert store.table_name == "my_custom_table"
        assert store.recreate_table is True
        assert store._client is None

    def test_invalid_table_name_raises(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        with pytest.raises(ValueError, match="Invalid table_name"):
            SupabaseGroongaDocumentStore(
                supabase_url="https://fake-project.supabase.co",
                table_name="bad-name; DROP TABLE users;",
            )

    def test_table_name_with_numbers_allowed(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_table_123",
        )
        assert store.table_name == "my_table_123"


class TestWarmUp:
    def test_warm_up_initializes_client(self, mock_groonga_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store._client is None
        store.warm_up()
        assert store._client is not None

    @pytest.mark.usefixtures("mock_groonga_store")
    def test_creates_the_table_and_the_pgroonga_index(self, mock_groonga_client):
        statements = [call.args[1]["query"] for call in mock_groonga_client.rpc.call_args_list]

        assert any("CREATE TABLE IF NOT EXISTS test_groonga_documents" in sql for sql in statements)
        assert any("USING pgroonga (content)" in sql for sql in statements)
        assert not any("DROP TABLE" in sql for sql in statements)

    def test_drops_the_table_first_when_recreate_table_is_set(self, mock_groonga_client, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake.supabase.co", table_name="docs", recreate_table=True
        )

        store.warm_up()

        statements = [call.args[1]["query"] for call in mock_groonga_client.rpc.call_args_list]
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


class TestSerde:
    def test_to_dict(self, mock_groonga_store):
        result = mock_groonga_store.to_dict()
        assert result["type"] == (
            "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
        )
        assert result["init_parameters"]["table_name"] == "test_groonga_documents"
        assert result["init_parameters"]["supabase_url"] == "https://fake-project.supabase.co"
        assert result["init_parameters"]["recreate_table"] is False

    def test_from_dict(self, mock_groonga_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        data = {
            "type": (
                "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
            ),
            "init_parameters": {
                "supabase_url": "https://fake-project.supabase.co",
                "supabase_key": {
                    "type": "env_var",
                    "env_vars": ["SUPABASE_SERVICE_KEY"],
                    "strict": True,
                },
                "table_name": "test_groonga_documents",
                "recreate_table": False,
            },
        }
        store = SupabaseGroongaDocumentStore.from_dict(data)
        assert store.table_name == "test_groonga_documents"
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None


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


_filter = SupabaseGroongaDocumentStore._filter_documents_in_memory


def _docs():
    return [
        Document(id="1", content="alpha", meta={"lang": "en", "score": 1, "active": True}),
        Document(id="2", content="beta", meta={"lang": "fr", "score": 2, "active": False}),
        Document(id="3", content="gamma", meta={"lang": "en", "score": 3}),
    ]


class TestFilterDocumentsInMemory:
    # --- flat (simple) condition — the previously broken path -----------------

    def test_flat_condition_eq(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "==", "value": "en"})
        assert [d.id for d in result] == ["1", "3"]

    def test_flat_condition_neq(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "!=", "value": "en"})
        assert [d.id for d in result] == ["2"]

    def test_flat_condition_in(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "in", "value": ["en", "de"]})
        assert [d.id for d in result] == ["1", "3"]

    def test_flat_condition_not_in(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "not in", "value": ["en"]})
        assert [d.id for d in result] == ["2"]

    # --- comparison operators -------------------------------------------------

    def test_gt(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": ">", "value": 1})
        assert [d.id for d in result] == ["2", "3"]

    def test_gte(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": ">=", "value": 2})
        assert [d.id for d in result] == ["2", "3"]

    def test_lt(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": "<", "value": 3})
        assert [d.id for d in result] == ["1", "2"]

    def test_lte(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": "<=", "value": 2})
        assert [d.id for d in result] == ["1", "2"]

    def test_gt_excludes_missing_field(self):
        # doc "3" has no "score" key — should be excluded since None > value is False
        docs = [
            Document(id="a", meta={"score": 5}),
            Document(id="b", meta={}),
        ]
        result = _filter(docs, {"field": "meta.score", "operator": ">", "value": 0})
        assert [d.id for d in result] == ["a"]

    # --- logical operators ----------------------------------------------------

    def test_and(self):
        result = _filter(
            _docs(),
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.score", "operator": ">", "value": 1},
                ],
            },
        )
        assert [d.id for d in result] == ["3"]

    def test_or(self):
        result = _filter(
            _docs(),
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "fr"},
                    {"field": "meta.score", "operator": "==", "value": 3},
                ],
            },
        )
        assert [d.id for d in result] == ["2", "3"]

    def test_not(self):
        result = _filter(
            _docs(),
            {"operator": "NOT", "conditions": [{"field": "meta.lang", "operator": "==", "value": "en"}]},
        )
        assert [d.id for d in result] == ["2"]

    def test_nested_and_inside_or(self):
        # (lang==en AND score==1) OR lang==fr
        result = _filter(
            _docs(),
            {
                "operator": "OR",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.lang", "operator": "==", "value": "en"},
                            {"field": "meta.score", "operator": "==", "value": 1},
                        ],
                    },
                    {"field": "meta.lang", "operator": "==", "value": "fr"},
                ],
            },
        )
        assert [d.id for d in result] == ["1", "2"]

    def test_empty_filters_returns_all(self):
        docs = _docs()
        assert _filter(docs, {}) == docs


class TestCountDocuments:
    def test_returns_the_reported_count(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.table.return_value.execute.return_value = MagicMock(count=7)

        assert mock_groonga_store.count_documents() == 7


class TestWriteDocuments:
    def test_overwrite_upserts_each_document(self, mock_groonga_store, mock_groonga_client):
        table = mock_groonga_client.table.return_value

        written = mock_groonga_store.write_documents(
            [Document(id="a", content="alpha", meta={"kind": "book"})], policy=DuplicatePolicy.OVERWRITE
        )

        assert written == 1
        table.upsert.assert_called_once_with({"id": "a", "content": "alpha", "meta": {"kind": "book"}, "score": None})

    def test_skip_leaves_an_existing_document_alone(self, mock_groonga_store, mock_groonga_client):
        table = mock_groonga_client.table.return_value
        table.execute.return_value = MagicMock(data=[{"id": "a"}])

        written = mock_groonga_store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.SKIP)

        assert written == 0
        table.insert.assert_not_called()

    def test_fail_raises_on_an_existing_document(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.table.return_value.execute.return_value = MagicMock(data=[{"id": "a"}])

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            mock_groonga_store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.FAIL)

    def test_writing_nothing_is_a_no_op(self, mock_groonga_store):
        assert mock_groonga_store.write_documents([]) == 0

    def test_rejects_something_that_is_not_a_list(self, mock_groonga_store):
        with pytest.raises(ValueError, match="expects a list of Document objects"):
            mock_groonga_store.write_documents("not a list")

    def test_rejects_a_list_holding_something_other_than_documents(self, mock_groonga_store):
        with pytest.raises(ValueError, match="expects Document objects"):
            mock_groonga_store.write_documents([Document(id="a"), "not a document"])


class TestDeleteAndUpdate:
    def test_deleting_nothing_is_a_no_op(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_store.delete_documents([])
        mock_groonga_client.table.return_value.delete.assert_not_called()

    def test_delete_documents_deletes_the_given_ids(self, mock_groonga_store, mock_groonga_client):
        table = mock_groonga_client.table.return_value

        mock_groonga_store.delete_documents(["a", "b"])

        table.delete.assert_called_once()
        table.in_.assert_called_once_with("id", ["a", "b"])

    def test_delete_all_documents_matches_every_row(self, mock_groonga_store, mock_groonga_client):
        table = mock_groonga_client.table.return_value

        mock_groonga_store.delete_all_documents()

        table.delete.assert_called_once()
        table.neq.assert_called_once_with("id", "")

    def test_delete_by_filter_returns_the_number_deleted(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.table.return_value.execute.return_value = MagicMock(
            data=[{"id": "a", "meta": {}}, {"id": "b", "meta": {}}]
        )

        assert mock_groonga_store.delete_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}) == 2

    def test_update_by_filter_merges_the_new_metadata(self, mock_groonga_store, mock_groonga_client):
        table = mock_groonga_client.table.return_value
        table.execute.return_value = MagicMock(data=[{"id": "a", "content": "alpha", "meta": {"kind": "book"}}])

        updated = mock_groonga_store.update_by_filter(
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
    def test_calls_the_search_function_with_the_query_and_table(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_store._groonga_retrieval(query="search text", top_k=5)

        mock_groonga_client.rpc.assert_called_with(
            "groonga_search", {"query_text": "search text", "table_name": "test_groonga_documents", "top_k": 5}
        )

    def test_converts_the_returned_rows(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.rpc.return_value.execute.return_value = MagicMock(
            data=[{"id": "a", "content": "alpha", "meta": {}, "score": 1.0}]
        )

        docs = mock_groonga_store._groonga_retrieval(query="alpha")

        assert [doc.id for doc in docs] == ["a"]

    def test_applies_filters_after_retrieval(self, mock_groonga_store, mock_groonga_client):
        mock_groonga_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {"id": "a", "content": "alpha", "meta": {"kind": "book"}},
                {"id": "b", "content": "beta", "meta": {"kind": "paper"}},
            ]
        )

        docs = mock_groonga_store._groonga_retrieval(
            query="alpha", filters={"field": "meta.kind", "operator": "==", "value": "book"}
        )

        assert [doc.id for doc in docs] == ["a"]


class TestAsyncPaths:
    async def test_count_documents_async(self, mock_async_groonga_store, mock_async_groonga_client):
        mock_async_groonga_client.table.return_value.execute = AsyncMock(return_value=MagicMock(count=3))

        assert await mock_async_groonga_store.count_documents_async() == 3

    async def test_write_documents_async_overwrite(self, mock_async_groonga_store, mock_async_groonga_client):
        written = await mock_async_groonga_store.write_documents_async(
            [Document(id="a", content="alpha")], policy=DuplicatePolicy.OVERWRITE
        )

        assert written == 1
        mock_async_groonga_client.table.return_value.upsert.assert_called_once()

    async def test_write_documents_async_fail_raises_on_a_duplicate(
        self, mock_async_groonga_store, mock_async_groonga_client
    ):
        mock_async_groonga_client.table.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "a"}]))

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            await mock_async_groonga_store.write_documents_async([Document(id="a")], policy=DuplicatePolicy.FAIL)

    async def test_delete_documents_async(self, mock_async_groonga_store, mock_async_groonga_client):
        table = mock_async_groonga_client.table.return_value

        await mock_async_groonga_store.delete_documents_async(["a"])

        table.in_.assert_called_once_with("id", ["a"])

    async def test_delete_by_filter_async(self, mock_async_groonga_store, mock_async_groonga_client):
        mock_async_groonga_client.table.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"id": "a", "meta": {}}])
        )

        assert (
            await mock_async_groonga_store.delete_by_filter_async({"field": "id", "operator": "==", "value": "a"}) == 1
        )

    async def test_update_by_filter_async_merges_the_new_metadata(
        self, mock_async_groonga_store, mock_async_groonga_client
    ):
        table = mock_async_groonga_client.table.return_value
        table.execute = AsyncMock(
            return_value=MagicMock(data=[{"id": "a", "content": "alpha", "meta": {"kind": "book"}}])
        )

        updated = await mock_async_groonga_store.update_by_filter_async(
            {"field": "id", "operator": "==", "value": "a"}, {"n": 1}
        )

        assert updated == 1
        assert table.upsert.call_args.args[0]["meta"] == {"kind": "book", "n": 1}

    async def test_groonga_retrieval_async_applies_filters_after_retrieval(
        self, mock_async_groonga_store, mock_async_groonga_client
    ):
        mock_async_groonga_client.rpc.return_value.execute = AsyncMock(
            return_value=MagicMock(
                data=[
                    {"id": "a", "content": "alpha", "meta": {"kind": "book"}},
                    {"id": "b", "content": "beta", "meta": {"kind": "paper"}},
                ]
            )
        )

        docs = await mock_async_groonga_store._groonga_retrieval_async(
            query="alpha", filters={"field": "meta.kind", "operator": "==", "value": "book"}
        )

        assert [doc.id for doc in docs] == ["a"]

    async def test_setup_table_async_requires_a_client(self, mock_async_groonga_store):
        with pytest.raises(RuntimeError, match="Async client not initialized"):
            await mock_async_groonga_store._setup_table_async()

    async def test_lazy_async_client_initialization(self, mock_async_groonga_client, monkeypatch):  # noqa: ARG002
        """Async client must be None at construction and set after the first async call."""
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store._async_client is None
        await store.count_documents_async()
        assert store._async_client is not None

    async def test_write_documents_async_empty(self, mock_async_groonga_store):
        written = await mock_async_groonga_store.write_documents_async([])
        assert written == 0

    async def test_delete_documents_async_empty(self, mock_async_groonga_store, mock_async_groonga_client):
        await mock_async_groonga_store.delete_documents_async([])
        mock_async_groonga_client.table.return_value.delete.assert_not_called()

    async def test_async_client_initialized_only_once(self, mock_async_groonga_client, monkeypatch):  # noqa: ARG002
        """_initialize_async_client must not replace the client on subsequent calls."""
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        await store.count_documents_async()
        first_client = store._async_client
        await store.count_documents_async()
        assert store._async_client is first_client
