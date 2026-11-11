# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.chroma.errors import ChromaDocumentStoreFilterError
from haystack_integrations.document_stores.chroma.filters import (
    _convert_filters,
    _create_where_document_filter,
    _parse_comparison_condition,
    _parse_logical_condition,
)


class TestParseComparisonCondition:
    @pytest.mark.parametrize(
        ("operator", "chroma_operator"),
        [
            ("==", "$eq"),
            ("!=", "$ne"),
            (">", "$gt"),
            (">=", "$gte"),
            ("<", "$lt"),
            ("<=", "$lte"),
            ("in", "$in"),
            ("not in", "$nin"),
        ],
    )
    def test_maps_every_comparison_operator(self, operator, chroma_operator):
        condition = {"field": "meta.year", "operator": operator, "value": 2024}

        assert _parse_comparison_condition(condition) == {"year": {chroma_operator: 2024}}

    def test_strips_the_meta_prefix_from_the_field_name(self):
        assert _parse_comparison_condition({"field": "meta.year", "operator": "==", "value": 1}) == {"year": {"$eq": 1}}

    @pytest.mark.parametrize(
        ("condition", "error_match"),
        [
            ({"operator": "==", "value": 1}, "'field' key missing"),
            ({"field": "meta.year", "value": 1}, "'operator' key missing"),
            ({"field": "meta.year", "operator": "=="}, "'value' key missing"),
            ({"field": "meta.year", "operator": "~=", "value": 1}, "Unknown operator"),
        ],
    )
    def test_malformed_conditions_raise(self, condition, error_match):
        with pytest.raises(ChromaDocumentStoreFilterError, match=error_match):
            _parse_comparison_condition(condition)


class TestParseLogicalCondition:
    def test_and(self):
        condition = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.kind", "operator": "==", "value": "book"},
            ],
        }

        assert _parse_logical_condition(condition) == {"$and": [{"year": {"$eq": 2024}}, {"kind": {"$eq": "book"}}]}

    @pytest.mark.parametrize(
        ("condition", "error_match"),
        [
            ({"conditions": []}, "'operator' key missing"),
            ({"operator": "AND"}, "'conditions' key missing"),
            ({"operator": "XOR", "conditions": []}, "Unknown operator"),
        ],
    )
    def test_malformed_conditions_raise(self, condition, error_match):
        with pytest.raises(ChromaDocumentStoreFilterError, match=error_match):
            _parse_logical_condition(condition)


class TestCreateWhereDocumentFilter:
    def test_a_content_field_becomes_the_document_filter(self):
        assert _create_where_document_filter("content", {"$contains": "text"}) == {"$contains": "text"}

    def test_a_metadata_field_produces_no_document_filter(self):
        assert _create_where_document_filter("year", {"$eq": 2024}) == {}

    def test_a_logical_operator_over_content_conditions_is_combined(self):
        value = [{"content": {"$contains": "first"}}, {"content": {"$contains": "second"}}]

        assert _create_where_document_filter("$and", value) == {
            "$and": [{"$contains": "first"}, {"$contains": "second"}]
        }


class TestConvertFilters:
    def test_a_metadata_filter_becomes_a_where_clause(self):
        chroma_filter = _convert_filters({"field": "meta.year", "operator": "==", "value": 2024})

        assert chroma_filter.where == {"year": {"$eq": 2024}}
        assert chroma_filter.ids == []
        assert chroma_filter.where_document is None

    def test_an_id_filter_becomes_an_ids_clause(self):
        chroma_filter = _convert_filters({"field": "id", "operator": "==", "value": "doc-1"})

        assert chroma_filter.ids == ["doc-1"]
        assert chroma_filter.where is None

    @pytest.mark.parametrize(
        "condition",
        [
            {"field": "id", "operator": "!=", "value": "doc-1"},
            {"field": "id", "operator": "==", "value": ""},
        ],
        ids=["wrong-operator", "empty-value"],
    )
    def test_an_id_filter_only_supports_equality_with_a_value(self, condition):
        with pytest.raises(ChromaDocumentStoreFilterError, match="id filter only supports"):
            _convert_filters(condition)

    def test_a_content_filter_becomes_a_where_document_clause(self):
        chroma_filter = _convert_filters({"field": "content", "operator": "contains", "value": "text"})

        assert chroma_filter.where_document == {"$contains": "text"}
        assert chroma_filter.where is None

    def test_a_logical_filter_over_metadata_becomes_a_where_clause(self):
        chroma_filter = _convert_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.year", "operator": "==", "value": 2024},
                    {"field": "meta.kind", "operator": "==", "value": "book"},
                ],
            }
        )

        assert chroma_filter.where == {"$and": [{"year": {"$eq": 2024}}, {"kind": {"$eq": "book"}}]}
        assert chroma_filter.where_document is None

    def test_a_logical_filter_over_content_becomes_a_where_document_clause(self):
        chroma_filter = _convert_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "content", "operator": "contains", "value": "first"},
                    {"field": "content", "operator": "contains", "value": "second"},
                ],
            }
        )

        assert chroma_filter.where_document == {"$and": [{"$contains": "first"}, {"$contains": "second"}]}

    def test_an_invalid_metadata_filter_is_rejected_by_chroma(self):
        with pytest.raises(ChromaDocumentStoreFilterError, match="Invalid 'metadata filter'"):
            _convert_filters({"operator": "AND", "conditions": []})
