# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.errors import FilterError

from haystack_integrations.document_stores.vespa.errors import VespaDocumentStoreFilterError
from haystack_integrations.document_stores.vespa.filters import _normalize_filters

from .conftest import DummyResponse


class TestNormalizeFilters:
    """Unit tests for the YQL filter normalizer."""

    def test_string_equality_filters_use_contains(self):
        yql_filter = _normalize_filters(
            {"field": "meta.category", "operator": "==", "value": "news"},
            content_field="content",
        )

        assert yql_filter == 'category contains "news"'

    def test_string_relational_filters_require_iso_dates(self):
        with pytest.raises(FilterError):
            _normalize_filters({"field": "meta.number", "operator": ">", "value": "1"}, content_field="content")

    @pytest.mark.parametrize(
        "leaf_filter, content_field, expected_yql",
        [
            ({"field": "meta.number", "operator": "==", "value": 100}, "content", "number = 100"),
            ({"field": "page", "operator": "==", "value": "100"}, "content", 'page contains "100"'),
            ({"field": "content", "operator": "==", "value": "hello"}, "body", 'body contains "hello"'),
            ({"field": "meta.featured", "operator": ">", "value": True}, "content", "featured > true"),
            ({"field": "meta.featured", "operator": "==", "value": True}, "content", "featured = true"),
            ({"field": "meta.featured", "operator": "==", "value": False}, "content", "featured = false"),
            ({"field": "meta.number", "operator": "==", "value": None}, "content", "number = null"),
            (
                {"field": "meta.category", "operator": "!=", "value": "news"},
                "content",
                '!( category contains "news" )',
            ),
            ({"field": "meta.number", "operator": "!=", "value": 2}, "content", "!( number = 2 )"),
            ({"field": "meta.number", "operator": ">", "value": 5}, "content", "number > 5"),
            ({"field": "meta.rating", "operator": ">", "value": 1.5}, "content", "rating > 1.5"),
            ({"field": "meta.number", "operator": ">=", "value": 5}, "content", "number >= 5"),
            ({"field": "meta.number", "operator": "<", "value": 5}, "content", "number < 5"),
            ({"field": "meta.number", "operator": "<=", "value": 5}, "content", "number <= 5"),
            ({"field": "meta.number", "operator": ">", "value": None}, "content", "false"),
            (
                {"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"},
                "content",
                'date > "1972-12-11T19:54:58"',
            ),
            ({"field": "meta.page", "operator": "in", "value": ["1", "2"]}, "content", 'page in ("1", "2")'),
            ({"field": "meta.page", "operator": "not in", "value": ["1"]}, "content", '!( page in ("1") )'),
            ({"field": "meta.name", "operator": "contains", "value": "foo"}, "content", 'name contains "foo"'),
            (
                {"field": "meta.name", "operator": "not contains", "value": "foo"},
                "content",
                '!( name contains "foo" )',
            ),
        ],
    )
    def test_leaf_filters_translate_to_yql(self, leaf_filter, content_field, expected_yql):
        assert _normalize_filters(leaf_filter, content_field=content_field) == expected_yql

    @pytest.mark.parametrize("empty_filters", [None, {}])
    def test_empty_filters_match_everything(self, empty_filters):
        assert _normalize_filters(empty_filters, content_field="content") == "true"

    @pytest.mark.parametrize(
        "logical_filter, expected_yql",
        [
            (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.name", "operator": "==", "value": "name_0"},
                    ],
                },
                '( number = 100 and name contains "name_0" )',
            ),
            (
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.name", "operator": "==", "value": "name_0"},
                    ],
                },
                '( number = 100 or name contains "name_0" )',
            ),
            (
                {"operator": "NOT", "conditions": [{"field": "meta.number", "operator": "==", "value": 100}]},
                "!( ( number = 100 ) )",
            ),
        ],
    )
    def test_logical_filters_translate_to_yql(self, logical_filter, expected_yql):
        assert _normalize_filters(logical_filter, content_field="content") == expected_yql

    @pytest.mark.parametrize(
        "invalid_filter, expected_error",
        [
            ({"field": "meta.x", "operator": "like", "value": "y"}, VespaDocumentStoreFilterError),
            ({"field": "meta.x", "operator": "in", "value": "not-a-list"}, VespaDocumentStoreFilterError),
            ({"field": "meta.x", "operator": "not in", "value": "not-a-list"}, VespaDocumentStoreFilterError),
            ({"operator": "AND", "conditions": []}, VespaDocumentStoreFilterError),
            ({"operator": "OR"}, VespaDocumentStoreFilterError),
            ({"operator": "NOT", "conditions": []}, VespaDocumentStoreFilterError),
            ({"operator": "==", "value": "y"}, VespaDocumentStoreFilterError),
            ({"field": "meta.x", "value": "y"}, VespaDocumentStoreFilterError),
            ({"field": "meta.x", "operator": "=="}, VespaDocumentStoreFilterError),
            ({"field": "meta.x", "operator": ">", "value": {"a": 1}}, FilterError),
            ({"field": "meta.x", "operator": ">", "value": [1, 2]}, FilterError),
            ({"field": "meta.x", "operator": ">", "value": "not-a-date"}, FilterError),
        ],
    )
    def test_invalid_filters_raise(self, invalid_filter, expected_error):
        with pytest.raises(expected_error):
            _normalize_filters(invalid_filter, content_field="content")

    def test_multi_condition_not_clause(self):
        yql_filter = _normalize_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            },
            content_field="content",
        )

        assert yql_filter == '!( ( number = 100 and name contains "name_0" ) )'


class TestFilterDocuments:
    """Unit tests for `VespaDocumentStore.filter_documents` against a mocked pyvespa app."""

    def test_filter_documents_returns_matching_results(self, mock_store):
        mock_store._app.query.return_value = DummyResponse(
            {
                "root": {
                    "children": [
                        {
                            "id": "id:docs:docs::1",
                            "relevance": 3.5,
                            "fields": {
                                "id": "1",
                                "content": "hello",
                                "embedding": [0.1, 0.2],
                                "category": "news",
                            },
                        }
                    ]
                }
            }
        )

        documents = mock_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "news"})

        assert len(documents) == 1
        assert documents[0].id == "1"
        assert documents[0].score == 3.5
        assert documents[0].meta == {"category": "news"}

    def test_filter_documents_with_none_value_uses_python_fallback(self, mock_store):
        """Vespa cannot express `field IS NULL`; the store falls back to a full scan + Python filter."""
        mock_store._query_documents = Mock(  # type:ignore[method-assign]
            return_value=[
                Document(id="1", content="with number", meta={"number": 1}),
                Document(id="2", content="without number"),
            ]
        )

        documents = mock_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})

        assert [document.id for document in documents] == ["2"]
        mock_store._query_documents.assert_called_once_with(where="true", top_k=mock_store.query_limit)

    def test_filter_documents_with_iso_date_comparison_uses_python_fallback(self, mock_store):
        """String relational comparison on ISO dates is evaluated client-side."""
        mock_store._query_documents = Mock(  # type:ignore[method-assign]
            return_value=[
                Document(id="1", content="old", meta={"date": "1969-07-21T20:17:40"}),
                Document(id="2", content="new", meta={"date": "1989-11-09T17:53:00"}),
            ]
        )

        documents = mock_store.filter_documents(
            filters={"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"}
        )

        assert [document.id for document in documents] == ["2"]
        mock_store._query_documents.assert_called_once_with(where="true", top_k=mock_store.query_limit)
