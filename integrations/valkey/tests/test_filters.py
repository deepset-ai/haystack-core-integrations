# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.errors import FilterError
from haystack.testing.document_store import FilterDocumentsTest

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore
from haystack_integrations.document_stores.valkey.filters import _normalize_filters, _validate_filters


@pytest.mark.integration
class TestFilters(FilterDocumentsTest):
    @pytest.fixture
    def document_store(self):
        store = ValkeyDocumentStore(
            index_name="test_filters",
            embedding_dim=768,
            metadata_fields={"number": int, "name": str, "page": str, "chapter": str, "date": str},
        )
        yield store
        try:
            store._client.flushdb()
            store.close()
        except Exception:  # noqa: S110
            pass

    def assert_documents_are_equal(self, received: list, expected: list):
        """Override to sort documents by ID before comparison since Valkey may return in different order."""
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)
        assert received_sorted == expected_sorted

    # Valkey does not support None/null value filtering for comparison operators
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        pytest.skip("Valkey does not support comparison operators with None values")

    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        pytest.skip("Valkey does not support comparison operators with None values")

    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        pytest.skip("Valkey does not support comparison operators with None values")

    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        pytest.skip("Valkey does not support comparison operators with None values")

    # Valkey only supports numeric comparisons on numeric fields, not on strings/dates/lists
    def test_comparison_greater_than_with_string(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_greater_than_with_list(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_greater_than_equal_with_string(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_greater_than_equal_with_list(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_with_string(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_with_list(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_equal_with_string(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")

    def test_comparison_less_than_equal_with_list(self, document_store, filterable_docs):
        pytest.skip("Valkey only supports numeric comparisons on numeric fields")


# Default supported fields for testing
DEFAULT_SUPPORTED_FIELDS = {
    "meta_category": "tag",
    "meta_score": "numeric",
    "meta_priority": "numeric",
    "meta_status": "tag",
    "meta_timestamp": "numeric",
}

filters_data = [
    # Basic TagField equality
    (
        {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]},
        "(@meta_category:{news})",
    ),
    # Basic NumericField comparison
    (
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.8}]},
        "(@meta_score:[0.8 +inf])",
    ),
    # Complex AND condition
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.score", "operator": ">=", "value": 0.5},
                {"field": "meta.priority", "operator": "<=", "value": 10},
            ],
        },
        "(@meta_category:{news} @meta_score:[0.5 +inf] @meta_priority:[-inf 10])",
    ),
    # OR condition
    (
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.category", "operator": "==", "value": "sports"},
            ],
        },
        "(@meta_category:{news} | @meta_category:{sports})",
    ),
    # IN operator for TagField
    (
        {"operator": "AND", "conditions": [{"field": "meta.status", "operator": "in", "value": ["active", "pending"]}]},
        "((@meta_status:{active} | @meta_status:{pending}))",
    ),
    # IN operator for NumericField
    (
        {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": "in", "value": [1, 2, 3]}]},
        "((@meta_priority:[1 1] | @meta_priority:[2 2] | @meta_priority:[3 3]))",
    ),
    # NOT EQUAL for TagField
    (
        {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "!=", "value": "spam"}]},
        "(-@meta_category:{spam})",
    ),
    # NOT IN for TagField
    (
        {
            "operator": "AND",
            "conditions": [{"field": "meta.status", "operator": "not in", "value": ["deleted", "archived"]}],
        },
        "(-@meta_status:{deleted | archived})",
    ),
    # Nested conditions
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.priority", "operator": ">", "value": 5},
                        {"field": "meta.score", "operator": ">=", "value": 0.9},
                    ],
                },
            ],
        },
        "(@meta_category:{news} (@meta_priority:[(5 +inf] | @meta_score:[0.9 +inf]))",
    ),
    # Range queries
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.timestamp", "operator": ">=", "value": 1609459200},
                {"field": "meta.timestamp", "operator": "<", "value": 1640995200},
            ],
        },
        "(@meta_timestamp:[1609459200 +inf] @meta_timestamp:[-inf (1640995200])",
    ),
]


@pytest.mark.parametrize("filters, expected", filters_data)
def test_normalize_filters(filters, expected):
    result = _normalize_filters(filters, DEFAULT_SUPPORTED_FIELDS)
    assert result == expected


def test_normalize_filters_invalid_operator():
    with pytest.raises(FilterError, match="Unknown logical operator 'INVALID'"):
        _normalize_filters({"operator": "INVALID", "conditions": []}, DEFAULT_SUPPORTED_FIELDS)


def test_normalize_filters_malformed():
    # Missing operator
    with pytest.raises(FilterError, match="'operator' key missing"):
        _normalize_filters({"conditions": []}, DEFAULT_SUPPORTED_FIELDS)

    # Missing conditions
    with pytest.raises(FilterError, match="'conditions' key missing"):
        _normalize_filters({"operator": "AND"}, DEFAULT_SUPPORTED_FIELDS)

    # Missing comparison field
    with pytest.raises(FilterError, match="'conditions' key missing"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"operator": "==", "value": "news"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )

    # Missing comparison operator
    with pytest.raises(FilterError, match="'operator' key missing"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "value": "news"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )

    # Missing comparison value
    with pytest.raises(FilterError, match="'value' key missing"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "=="}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_unsupported_field():
    with pytest.raises(FilterError, match="Field 'meta_unsupported' is not supported for filtering"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.unsupported", "operator": "==", "value": "test"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_unsupported_operator_for_tag_field():
    with pytest.raises(FilterError, match="Operator '>' not supported for tag field"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": ">", "value": "news"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_invalid_value_type_for_tag_field():
    with pytest.raises(FilterError, match="TagField 'meta_category' requires string value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": 123}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_invalid_value_type_for_numeric_field():
    with pytest.raises(FilterError, match="NumericField 'meta_score' requires numeric value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": "invalid"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_invalid_in_operator_value():
    with pytest.raises(FilterError, match="'in' operator requires a list value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "in", "value": "not_a_list"}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_invalid_list_values_for_tag_field():
    with pytest.raises(FilterError, match="TagField 'meta_category' requires string values in list"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "in", "value": ["valid", 123]}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_invalid_list_values_for_numeric_field():
    with pytest.raises(FilterError, match="NumericField 'meta_priority' requires numeric values in list"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": "in", "value": [1, "invalid"]}]},
            DEFAULT_SUPPORTED_FIELDS,
        )


def test_validate_filters_valid():
    # Should not raise any exception
    _validate_filters({"operator": "AND", "conditions": []})
    _validate_filters(None)


def test_validate_filters_invalid():
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"invalid": "filter"})


def test_special_character_escaping():
    # Test that special characters are properly escaped
    result = _normalize_filters(
        {
            "operator": "AND",
            "conditions": [{"field": "meta.category", "operator": "==", "value": "test-value.with:special@chars"}],
        },
        DEFAULT_SUPPORTED_FIELDS,
    )
    expected = "(@meta_category:{test\\-value\\.with\\:special\\@chars})"
    assert result == expected


def test_direct_comparison_condition():
    # Test single comparison condition without logical wrapper
    result = _normalize_filters({"field": "meta.category", "operator": "==", "value": "news"}, DEFAULT_SUPPORTED_FIELDS)
    assert result == "@meta_category:{news}"


def test_numeric_equality():
    result = _normalize_filters(
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": "==", "value": 0.5}]},
        DEFAULT_SUPPORTED_FIELDS,
    )
    assert result == "(@meta_score:[0.5 0.5])"


def test_numeric_not_equal():
    result = _normalize_filters(
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": "!=", "value": 0.5}]},
        DEFAULT_SUPPORTED_FIELDS,
    )
    assert result == "(-@meta_score:[0.5 0.5])"


def test_filters_must_be_dict():
    with pytest.raises(FilterError, match="Filters must be a dictionary"):
        _normalize_filters("invalid", DEFAULT_SUPPORTED_FIELDS)
