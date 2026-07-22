# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import FilterDocumentsTest
from psycopg.adapt import Transformer
from psycopg.sql import SQL, Composed
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

from haystack_integrations.document_stores.pgvector.filters import (
    FilterError,
    _convert_filters_to_where_clause_and_params,
    _parse_comparison_condition,
    _parse_logical_condition,
    _treat_meta_field,
    _treat_meta_field_as_jsonb,
    _validate_filters,
)


def _render(composed: Composed) -> str:
    """Render a psycopg Composed object to a plain SQL string for assertions."""
    return composed.as_string(Transformer())


@pytest.mark.integration
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        This overrides the default assert_documents_are_equal from FilterDocumentsTest.
        It is needed because the embeddings are not exactly the same when they are retrieved from Postgres.
        """

        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            # we first compare the embeddings approximately
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

            received_doc_no_embedding = replace(received_doc, embedding=None)
            expected_doc_no_embedding = replace(expected_doc, embedding=None)
            assert received_doc_no_embedding == expected_doc_no_embedding

    @pytest.mark.skip(reason="NOT operator is not supported in PgvectorDocumentStore")
    def test_not_operator(self, document_store, filterable_docs): ...

    def test_like_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "like", "value": "%Foo%"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(result, [d for d in filterable_docs if "Foo" in d.content])

    def test_like_operator_startswith(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "like", "value": "Foo%"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.content.startswith("Foo")])

    def test_like_operator_nb_chars(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "like", "value": "A Foobar Document__"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.content.startswith("A Foobar Document") and len(d.content) == (len("A Foobar Document") + 2))
            ],
        )

    def test_not_like_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "not like", "value": "%Foo%"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(result, [d for d in filterable_docs if "Foo" not in d.content])

    def test_not_like_operator_startswith(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "not like", "value": "Foo%"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(result, [d for d in filterable_docs if not d.content.startswith("Foo")])

    def test_not_like_operator_nb_chars(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {"field": "content", "operator": "not like", "value": "A Foobar Document__"}
        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if not (d.content.startswith("A Foobar Document") and len(d.content) == (len("A Foobar Document") + 2))
            ],
        )

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": "90"},
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                    ],
                },
            ],
        }

        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
                or (d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion")
            ],
        )

    @staticmethod
    def _docs_with_array_meta():
        return [
            Document(content="doc with tag1 and tag2", meta={"tags": ["tag1", "tag2"]}),
            Document(content="doc with tag1 and tag3", meta={"tags": ["tag1", "tag3"]}),
            Document(content="doc with tag4", meta={"tags": ["tag4"]}),
        ]

    def test_array_contains_operator(self, document_store):
        docs = self._docs_with_array_meta()
        document_store.write_documents(docs)

        # array_contains requires ALL listed values to be present in the meta array
        result = document_store.filter_documents(
            filters={"field": "meta.tags", "operator": "array_contains", "value": ["tag1", "tag2"]}
        )
        assert {doc.content for doc in result} == {"doc with tag1 and tag2"}

    def test_array_contains_operator_single_value(self, document_store):
        docs = self._docs_with_array_meta()
        document_store.write_documents(docs)

        result = document_store.filter_documents(
            filters={"field": "meta.tags", "operator": "array_contains", "value": ["tag1"]}
        )
        assert {doc.content for doc in result} == {"doc with tag1 and tag2", "doc with tag1 and tag3"}

    def test_array_overlaps_operator(self, document_store):
        docs = self._docs_with_array_meta()
        document_store.write_documents(docs)

        # array_overlaps requires AT LEAST ONE listed value to be present in the meta array
        result = document_store.filter_documents(
            filters={"field": "meta.tags", "operator": "array_overlaps", "value": ["tag1", "tag3"]}
        )
        assert {doc.content for doc in result} == {"doc with tag1 and tag2", "doc with tag1 and tag3"}

    def test_array_overlaps_operator_no_match(self, document_store):
        docs = self._docs_with_array_meta()
        document_store.write_documents(docs)

        result = document_store.filter_documents(
            filters={"field": "meta.tags", "operator": "array_overlaps", "value": ["nonexistent"]}
        )
        assert result == []

    def test_array_operators_combined(self, document_store):
        docs = self._docs_with_array_meta()
        document_store.write_documents(docs)

        result = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.tags", "operator": "array_contains", "value": ["tag1"]},
                    {"field": "meta.tags", "operator": "array_overlaps", "value": ["tag3"]},
                ],
            }
        )
        assert {doc.content for doc in result} == {"doc with tag1 and tag3"}


def test_treat_meta_field():
    cast_integer = SQL("(") + SQL("meta->>") + SQLLiteral("number") + SQL(")::integer")
    no_cast_name = SQL("meta->>") + SQLLiteral("name")
    cast_real = SQL("(") + SQL("meta->>") + SQLLiteral("number") + SQL(")::real")
    cast_boolean = SQL("(") + SQL("meta->>") + SQLLiteral("bool") + SQL(")::boolean")

    assert _treat_meta_field(field="meta.number", value=9) == cast_integer
    assert _treat_meta_field(field="meta.number", value=[1, 2, 3]) == cast_integer
    assert _treat_meta_field(field="meta.name", value="my_name") == no_cast_name
    assert _treat_meta_field(field="meta.name", value=["my_name"]) == no_cast_name
    assert _treat_meta_field(field="meta.number", value=1.1) == cast_real
    assert _treat_meta_field(field="meta.number", value=[1.1, 2.2]) == cast_real
    assert _treat_meta_field(field="meta.bool", value=True) == cast_boolean
    assert _treat_meta_field(field="meta.bool", value=[True, False]) == cast_boolean


def test_treat_meta_field_sql_injection_is_safely_escaped():
    # SQL injection attempts are safely escaped by SQLLiteral rather than rejected
    result = _treat_meta_field(field="meta.name' OR 1=1 --", value="x")
    assert isinstance(result, Composed)
    assert result == SQL("meta->>") + SQLLiteral("name' OR 1=1 --")


def test_treat_meta_field_as_jsonb():
    # array operators access the meta value as JSONB (->) without a cast, unlike _treat_meta_field (->>)
    assert _treat_meta_field_as_jsonb(field="meta.tags") == SQL("meta->") + SQLLiteral("tags")


def test_treat_meta_field_as_jsonb_sql_injection_is_safely_escaped():
    result = _treat_meta_field_as_jsonb(field="meta.tags' OR 1=1 --")
    assert isinstance(result, Composed)
    assert result == SQL("meta->") + SQLLiteral("tags' OR 1=1 --")


def test_array_contains_condition():
    condition = {"field": "meta.tags", "operator": "array_contains", "value": ["tag1", "tag2"]}
    query, values = _parse_comparison_condition(condition)

    assert _render(query) == "meta->'tags' @> %s"
    # the value is wrapped as a single Jsonb param holding the original list
    assert len(values) == 1
    assert isinstance(values[0], Jsonb)
    assert values[0].obj == ["tag1", "tag2"]


def test_array_overlaps_condition():
    condition = {"field": "meta.tags", "operator": "array_overlaps", "value": ["tag1", "tag2"]}
    query, values = _parse_comparison_condition(condition)

    assert _render(query) == "meta->'tags' ?| %s"
    # the list is passed through so psycopg adapts it to a Postgres text[] (same wrapping as 'in')
    assert values == [[["tag1", "tag2"]]]


def test_array_contains_non_list_value_raises():
    condition = {"field": "meta.tags", "operator": "array_contains", "value": "tag1"}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_array_overlaps_non_list_value_raises():
    condition = {"field": "meta.tags", "operator": "array_overlaps", "value": "tag1"}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_array_overlaps_non_string_elements_raises():
    condition = {"field": "meta.tags", "operator": "array_overlaps", "value": [1, 2]}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_convert_filters_array_contains():
    filters = {"field": "meta.tags", "operator": "array_contains", "value": ["tag1", "tag2"]}
    where_clause, params = _convert_filters_to_where_clause_and_params(filters)

    assert _render(where_clause) == " WHERE meta->'tags' @> %s"
    assert len(params) == 1
    assert isinstance(params[0], Jsonb)
    assert params[0].obj == ["tag1", "tag2"]


def test_convert_filters_array_overlaps():
    filters = {"field": "meta.tags", "operator": "array_overlaps", "value": ["tag1", "tag2"]}
    where_clause, params = _convert_filters_to_where_clause_and_params(filters)

    assert _render(where_clause) == " WHERE meta->'tags' ?| %s"
    assert params == ([["tag1", "tag2"]],)


def test_comparison_condition_missing_operator():
    condition = {"field": "meta.type", "value": "article"}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_comparison_condition_missing_value():
    condition = {"field": "meta.type", "operator": "=="}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_comparison_condition_unknown_operator():
    condition = {"field": "meta.type", "operator": "unknown", "value": "article"}
    with pytest.raises(FilterError):
        _parse_comparison_condition(condition)


def test_logical_condition_missing_operator():
    condition = {"conditions": []}
    with pytest.raises(FilterError):
        _parse_logical_condition(condition)


def test_logical_condition_missing_conditions():
    condition = {"operator": "AND"}
    with pytest.raises(FilterError):
        _parse_logical_condition(condition)


def test_logical_condition_unknown_operator():
    condition = {"operator": "unknown", "conditions": []}
    with pytest.raises(FilterError):
        _parse_logical_condition(condition)


def test_logical_condition_nested():
    condition = {
        "operator": "AND",
        "conditions": [
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.domain", "operator": "!=", "value": "science"},
                    {"field": "meta.chapter", "operator": "in", "value": ["intro", "conclusion"]},
                ],
            },
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.number", "operator": ">=", "value": 90},
                    {"field": "meta.author", "operator": "not in", "value": ["John", "Jane"]},
                ],
            },
        ],
    }
    query, values = _parse_logical_condition(condition)
    assert isinstance(query, Composed)

    expected_sql = (
        "("
        "(meta->>'domain' IS DISTINCT FROM %s OR meta->>'chapter' = ANY(%s))"
        " AND "
        "((meta->>'number')::integer >= %s OR meta->>'author' IS NULL OR meta->>'author' != ALL(%s))"
        ")"
    )
    assert _render(query) == expected_sql
    assert values == ["science", [["intro", "conclusion"]], 90, [["John", "Jane"]]]


def test_convert_filters_to_where_clause_and_params():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.number", "operator": "==", "value": 100},
            {"field": "meta.chapter", "operator": "==", "value": "intro"},
        ],
    }
    where_clause, params = _convert_filters_to_where_clause_and_params(filters)

    expected_sql = " WHERE ((meta->>'number')::integer = %s AND meta->>'chapter' = %s)"
    assert _render(where_clause) == expected_sql
    assert params == (100, "intro")


def test_convert_filters_to_where_clause_and_params_handle_null():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.number", "operator": "==", "value": None},
            {"field": "meta.chapter", "operator": "==", "value": "intro"},
        ],
    }
    where_clause, params = _convert_filters_to_where_clause_and_params(filters)

    expected_sql = " WHERE (meta->>'number' IS NULL AND meta->>'chapter' = %s)"
    assert _render(where_clause) == expected_sql
    assert params == ("intro",)


def test_convert_filters_to_where_clause_and_params_value_equal_to_no_value_sentinel():
    """A legitimate equality filter whose value is the literal string the internal
    NO_VALUE sentinel used to be ("no_value") must not be dropped from params --
    the query still has a placeholder for it."""
    filters = {"field": "meta.tag", "operator": "==", "value": "no_value"}
    where_clause, params = _convert_filters_to_where_clause_and_params(filters)

    expected_sql = " WHERE meta->>'tag' = %s"
    assert _render(where_clause) == expected_sql
    assert params == ("no_value",)


def test_validate_filters():
    filters = {"field": "meta.number", "operator": "==", "value": 100}
    _validate_filters(filters)

    my_list = ["a", "nice", "list"]
    with pytest.raises(TypeError):
        _validate_filters(my_list)

    invalid_filters = {"field": "meta.number", "value": "100"}
    with pytest.raises(ValueError):
        _validate_filters(invalid_filters)
