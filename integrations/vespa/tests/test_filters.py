# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.errors import FilterError

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
