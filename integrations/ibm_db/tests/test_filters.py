# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.ibm_db.filters import FilterTranslator, _is_iso_date


def _translate(filters):
    """Helper to translate filters and return SQL + params."""
    params = []
    sql = FilterTranslator().translate(filters, params)
    return sql, params


def test_equality():
    """Test equality operator."""
    sql, params = _translate({"field": "meta.author", "operator": "==", "value": "Alice"})
    assert "JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.author' RETURNING VARCHAR(1000))" in sql
    assert "= ?" in sql
    assert params == ["Alice"]


def test_inequality():
    """Test inequality operator."""
    sql, params = _translate({"field": "meta.status", "operator": "!=", "value": "draft"})
    assert "!= ?" in sql
    assert params == ["draft"]


def test_greater_than():
    """Test greater than operator."""
    sql, params = _translate({"field": "meta.year", "operator": ">", "value": 2020})
    assert "> ?" in sql
    assert params == [2020]


def test_greater_than_or_equal():
    """Test greater than or equal operator."""
    sql, params = _translate({"field": "meta.rating", "operator": ">=", "value": 4.5})
    assert ">= ?" in sql
    assert params == [4.5]


def test_less_than():
    """Test less than operator."""
    sql, params = _translate({"field": "meta.count", "operator": "<", "value": 100})
    assert "< ?" in sql
    assert params == [100]


def test_less_than_or_equal():
    """Test less than or equal operator."""
    sql, params = _translate({"field": "meta.score", "operator": "<=", "value": 50})
    assert "<= ?" in sql
    assert params == [50]


def test_in_operator():
    """Test IN operator."""
    sql, params = _translate({"field": "meta.lang", "operator": "in", "value": ["en", "de", "fr"]})
    assert "IN (?, ?, ?)" in sql
    assert params == ["en", "de", "fr"]


def test_not_in_operator():
    """Test NOT IN operator."""
    sql, params = _translate({"field": "meta.lang", "operator": "not in", "value": ["xx", "yy"]})
    assert "NOT IN (?, ?)" in sql
    assert "IS NULL OR" in sql
    assert params == ["xx", "yy"]


def test_and_logical():
    """Test AND logical operator."""
    sql, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.author", "operator": "==", "value": "Alice"},
                {"field": "meta.year", "operator": ">", "value": 2020},
            ],
        }
    )
    assert sql.startswith("(")
    assert " AND " in sql
    assert len(params) == 2
    assert params == ["Alice", 2020]


def test_or_logical():
    """Test OR logical operator."""
    sql, params = _translate(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": "x"},
                {"field": "meta.b", "operator": "==", "value": "y"},
            ],
        }
    )
    assert " OR " in sql
    assert len(params) == 2


def test_not_logical():
    """Test NOT logical operator."""
    sql, params = _translate(
        {
            "operator": "NOT",
            "conditions": [{"field": "meta.hidden", "operator": "==", "value": True}],
        }
    )
    assert sql.startswith("(NOT ")
    assert len(params) == 1


def test_nested_and_or():
    """Test nested AND/OR conditions."""
    sql, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.type", "operator": "==", "value": "article"},
                        {"field": "meta.type", "operator": "==", "value": "blog"},
                    ],
                },
            ],
        }
    )
    assert " AND " in sql
    assert " OR " in sql
    assert len(params) == 3
    assert params == ["en", "article", "blog"]


def test_id_field_maps_to_id_column():
    """Test that 'id' field maps directly to id column."""
    sql, params = _translate({"field": "id", "operator": "==", "value": "doc123"})
    assert "id = ?" in sql
    assert params == ["doc123"]


def test_content_field_maps_to_content_column():
    """Test that 'content' field maps directly to content column."""
    sql, params = _translate({"field": "content", "operator": "==", "value": "hello"})
    assert "content = ?" in sql
    assert params == ["hello"]


def test_nested_meta_key():
    """Test nested metadata key path."""
    sql, params = _translate({"field": "meta.author.city", "operator": "==", "value": "NYC"})
    assert "'$.author.city'" in sql
    assert params == ["NYC"]


def test_field_without_meta_prefix():
    """Test field without 'meta.' prefix is treated as metadata."""
    sql, params = _translate({"field": "language", "operator": "==", "value": "python"})
    assert "'$.language'" in sql
    assert params == ["python"]


def test_dollar_operators():
    """Test MongoDB-style $ operators."""
    sql, params = _translate({"field": "meta.count", "operator": "$gt", "value": 5})
    assert "> ?" in sql
    assert params == [5]


def test_complex_nested_conditions():
    """Test complex nested filter conditions."""
    sql, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "tech"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.rating", "operator": ">=", "value": 4},
                        {"field": "meta.featured", "operator": "==", "value": True},
                    ],
                },
                {"field": "meta.lang", "operator": "in", "value": ["en", "de"]},
            ],
        }
    )
    assert " AND " in sql
    assert " OR " in sql
    assert "IN (?, ?)" in sql
    assert len(params) == 5
    assert params == ["tech", 4, "true", "en", "de"]


def test_boolean_value_normalized_to_json_string():
    """Booleans are bound as the lowercase JSON strings JSON_VALUE returns, not Python bools."""
    _, true_params = _translate({"field": "meta.featured", "operator": "==", "value": True})
    assert true_params == ["true"]

    _, false_params = _translate({"field": "meta.featured", "operator": "!=", "value": False})
    assert false_params == ["false"]

    _, in_params = _translate({"field": "meta.flags", "operator": "in", "value": [True, False]})
    assert in_params == ["true", "false"]


# Made with Bob


class TestFilterTranslatorErrorHandling:
    """Test error handling and edge cases for FilterTranslator."""

    def test_empty_conditions_list_and_operator(self):
        """Test that AND operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "AND", "conditions": []})

    def test_empty_conditions_list_or_operator(self):
        """Test that OR operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "OR", "conditions": []})

    def test_empty_conditions_list_not_operator(self):
        """Test that NOT operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "NOT", "conditions": []})

    def test_empty_conditions_list_dollar_and_operator(self):
        """Test that $and operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "$and", "conditions": []})

    def test_empty_conditions_list_dollar_or_operator(self):
        """Test that $or operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "$or", "conditions": []})

    def test_empty_conditions_list_dollar_not_operator(self):
        """Test that $not operator with empty conditions raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty 'conditions' list"):
            _translate({"operator": "$not", "conditions": []})

    def test_not_operator_with_multiple_conditions(self):
        """NOT with multiple conditions negates their AND (Haystack semantics)."""
        sql, params = _translate(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.a", "operator": "==", "value": "x"},
                    {"field": "meta.b", "operator": "==", "value": "y"},
                ],
            }
        )
        assert sql.startswith("(NOT (")
        assert " AND " in sql
        assert params == ["x", "y"]

    def test_dollar_not_operator_with_multiple_conditions(self):
        """$not with multiple conditions negates their AND (Haystack semantics)."""
        sql, params = _translate(
            {
                "operator": "$not",
                "conditions": [
                    {"field": "meta.a", "operator": "==", "value": "x"},
                    {"field": "meta.b", "operator": "==", "value": "y"},
                ],
            }
        )
        assert sql.startswith("(NOT (")
        assert " AND " in sql
        assert params == ["x", "y"]

    def test_missing_field_key(self):
        """Test that missing 'field' key raises FilterError."""
        with pytest.raises(FilterError, match="must include a 'field' key"):
            _translate({"operator": "==", "value": "test"})

    def test_empty_field_key(self):
        """Test that empty 'field' key raises FilterError."""
        with pytest.raises(FilterError, match="must include a 'field' key"):
            _translate({"field": "", "operator": "==", "value": "test"})

    def test_missing_operator_key(self):
        """Test that missing 'operator' key raises FilterError."""
        with pytest.raises(FilterError, match="must include an 'operator' key"):
            _translate({"field": "meta.author", "value": "Alice"})

    def test_empty_operator_key(self):
        """Test that empty 'operator' key raises FilterError."""
        with pytest.raises(FilterError, match="must include an 'operator' key"):
            _translate({"field": "meta.author", "operator": "", "value": "Alice"})

    def test_invalid_operator(self):
        """Test that unsupported operator raises FilterError."""
        with pytest.raises(FilterError, match="Unsupported filter operator"):
            _translate({"field": "meta.author", "operator": "INVALID", "value": "Alice"})

    def test_in_operator_with_empty_list(self):
        """Test that IN operator with empty list raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "in", "value": []})

    def test_in_operator_with_non_list(self):
        """Test that IN operator with non-list value raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "in", "value": "en"})

    def test_not_in_operator_with_empty_list(self):
        """Test that NOT IN operator with empty list raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "not in", "value": []})

    def test_not_in_operator_with_non_list(self):
        """Test that NOT IN operator with non-list value raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "not in", "value": "en"})

    def test_dollar_in_operator_with_empty_list(self):
        """Test that $in operator with empty list raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "$in", "value": []})

    def test_dollar_nin_operator_with_empty_list(self):
        """Test that $nin operator with empty list raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "$nin", "value": []})

    def test_dollar_in_operator_with_non_list(self):
        """Test that $in operator with non-list value raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "$in", "value": "en"})

    def test_dollar_nin_operator_with_non_list(self):
        """Test that $nin operator with non-list value raises FilterError."""
        with pytest.raises(FilterError, match="requires a non-empty list value"):
            _translate({"field": "meta.lang", "operator": "$nin", "value": "en"})


class TestFilterTranslatorEdgeCases:
    """Test edge cases for FilterTranslator."""

    def test_filter_with_none_value(self):
        """Test filter with None value maps to an IS NULL predicate (no bound param)."""
        sql, params = _translate({"field": "meta.optional", "operator": "==", "value": None})
        assert sql.endswith("IS NULL")
        assert params == []

    def test_filter_with_empty_string_value(self):
        """Test filter with empty string value."""
        sql, params = _translate({"field": "meta.text", "operator": "==", "value": ""})
        assert "= ?" in sql
        assert params == [""]

    def test_filter_with_zero_value(self):
        """Test filter with zero value."""
        sql, params = _translate({"field": "meta.count", "operator": "==", "value": 0})
        assert "= ?" in sql
        assert params == [0]

    def test_filter_with_false_value(self):
        """Test filter with False boolean value."""
        sql, params = _translate({"field": "meta.active", "operator": "==", "value": False})
        assert "= ?" in sql
        assert params == ["false"]

    def test_filter_with_negative_number(self):
        """Test filter with negative number."""
        sql, params = _translate({"field": "meta.temperature", "operator": "<", "value": -10})
        assert "< ?" in sql
        assert params == [-10]

    def test_filter_with_float_value(self):
        """Test filter with float value."""
        sql, params = _translate({"field": "meta.rating", "operator": ">=", "value": 4.75})
        assert ">= ?" in sql
        assert params == [4.75]

    def test_in_operator_with_mixed_types(self):
        """Test IN operator with mixed types in list."""
        sql, params = _translate({"field": "meta.values", "operator": "in", "value": [1, "two", 3.0, True]})
        assert "IN (?, ?, ?, ?)" in sql
        # Booleans are normalized to JSON strings ('true'/'false') for DB2 compatibility
        assert params == [1, "two", 3.0, "true"]

    def test_deeply_nested_logical_operators(self):
        """Test deeply nested logical operators."""
        sql, params = _translate(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.a", "operator": "==", "value": 1},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.b", "operator": "==", "value": 2},
                            {
                                "operator": "AND",
                                "conditions": [
                                    {"field": "meta.c", "operator": "==", "value": 3},
                                    {"field": "meta.d", "operator": "==", "value": 4},
                                ],
                            },
                        ],
                    },
                ],
            }
        )
        assert " AND " in sql
        assert " OR " in sql
        assert len(params) == 4
        assert params == [1, 2, 3, 4]

    def test_not_operator_with_nested_and(self):
        """Test NOT operator with nested AND condition."""
        sql, params = _translate(
            {
                "operator": "NOT",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.x", "operator": "==", "value": "a"},
                            {"field": "meta.y", "operator": "==", "value": "b"},
                        ],
                    }
                ],
            }
        )
        assert sql.startswith("(NOT ")
        assert " AND " in sql
        assert len(params) == 2

    def test_dollar_operators_all_variants(self):
        """Test all MongoDB-style $ operators."""
        test_cases = [
            ("$eq", "="),
            ("$ne", "!="),
            ("$gt", ">"),
            ("$gte", ">="),
            ("$lt", "<"),
            ("$lte", "<="),
        ]
        for dollar_op, sql_op in test_cases:
            sql, params = _translate({"field": "meta.value", "operator": dollar_op, "value": 10})
            assert sql_op in sql
            assert params == [10]

    def test_embedding_field_direct_mapping(self):
        """Test that 'embedding' field maps directly to embedding column."""
        sql, params = _translate({"field": "embedding", "operator": "==", "value": "test"})
        assert "embedding = ?" in sql
        assert params == ["test"]

    def test_very_long_field_path(self):
        """Test filter with very long nested field path."""
        sql, params = _translate({"field": "meta.level1.level2.level3.level4.value", "operator": "==", "value": "deep"})
        assert "'$.level1.level2.level3.level4.value'" in sql
        assert params == ["deep"]

    def test_field_with_special_characters(self):
        """Test field with special characters in name."""
        sql, params = _translate({"field": "meta.field-name_123", "operator": "==", "value": "test"})
        assert "'$.field-name_123'" in sql
        assert params == ["test"]


class TestIsIsoDateFunction:
    """Test the _is_iso_date helper function."""

    def test_valid_iso_date_string(self):
        """Test that valid ISO date string returns True."""
        assert _is_iso_date("2023-12-25T10:30:00") is True

    def test_valid_iso_date_with_timezone(self):
        """Test that valid ISO date with timezone returns True."""
        assert _is_iso_date("2023-12-25T10:30:00+05:30") is True

    def test_valid_iso_date_with_z_timezone(self):
        """Test that valid ISO date with Z timezone returns True."""
        assert _is_iso_date("2023-12-25T10:30:00Z") is True

    def test_valid_iso_date_with_microseconds(self):
        """Test that valid ISO date with microseconds returns True."""
        assert _is_iso_date("2023-12-25T10:30:00.123456") is True

    def test_date_only_string(self):
        """Test that date-only string returns True."""
        assert _is_iso_date("2023-12-25") is True

    def test_invalid_date_string(self):
        """Test that invalid date string returns False."""
        assert _is_iso_date("not-a-date") is False

    def test_invalid_date_format(self):
        """Test that invalid date format returns False."""
        assert _is_iso_date("25/12/2023") is False

    def test_non_string_integer(self):
        """Test that non-string integer returns False."""
        assert _is_iso_date(12345) is False

    def test_non_string_none(self):
        """Test that None returns False."""
        assert _is_iso_date(None) is False

    def test_non_string_list(self):
        """Test that list returns False."""
        assert _is_iso_date(["2023-12-25"]) is False

    def test_empty_string(self):
        """Test that empty string returns False."""
        assert _is_iso_date("") is False

    def test_partial_iso_date(self):
        """Test that partial ISO date returns False."""
        assert _is_iso_date("2023-12") is False


# Made with Bob
