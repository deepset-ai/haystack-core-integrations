# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import weaviate
from haystack.errors import FilterError
from weaviate.collections.classes.filters import Filter

from haystack_integrations.document_stores.weaviate._filters import (
    _match_no_document,
    _parse_comparison_condition,
    _parse_logical_condition,
    convert_filters,
    validate_filters,
)

_property = weaviate.classes.query.Filter.by_property

ORDERING_OPERATORS = [">", ">=", "<", "<="]


def described(weaviate_filter):
    """
    Describe a Weaviate filter tree as nested tuples.

    Weaviate's composite filters have no `__eq__`, so comparing descriptions is what
    lets a test state the whole expected filter instead of poking at attributes.
    """
    if hasattr(weaviate_filter, "filters"):
        return (weaviate_filter.operator.value, [described(f) for f in weaviate_filter.filters])
    return (weaviate_filter.target, weaviate_filter.operator.value, weaviate_filter.value)


def comparison(condition):
    return described(_parse_comparison_condition(condition))


class TestValidateFilters:
    def test_rejects_a_filter_with_neither_an_operator_nor_conditions(self):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            validate_filters({"field": "meta.year"})


class TestComparisonOperators:
    def test_equal(self):
        assert comparison({"field": "meta.year", "operator": "==", "value": 2024}) == described(
            _property("year").equal(2024)
        )

    def test_equal_to_none_becomes_an_is_none_check(self):
        assert comparison({"field": "meta.year", "operator": "==", "value": None}) == described(
            _property("year").is_none(True)
        )

    def test_equal_normalizes_dates(self):
        assert comparison({"field": "meta.date", "operator": "==", "value": "2024-01-15"}) == described(
            _property("date").equal("2024-01-15T00:00:00.000000Z")
        )

    def test_not_equal_also_matches_documents_missing_the_field(self):
        # Weaviate excludes documents without the property, so they need an explicit clause.
        assert comparison({"field": "meta.year", "operator": "!=", "value": 2024}) == described(
            _property("year").not_equal(2024) | _property("year").is_none(True)
        )

    def test_not_equal_to_none_becomes_an_is_none_check(self):
        assert comparison({"field": "meta.year", "operator": "!=", "value": None}) == described(
            _property("year").is_none(False)
        )

    @pytest.mark.parametrize(
        ("operator", "expected_weaviate_operator"),
        [
            (">", "GreaterThan"),
            (">=", "GreaterThanEqual"),
            ("<", "LessThan"),
            ("<=", "LessThanEqual"),
        ],
    )
    def test_ordering_operators(self, operator, expected_weaviate_operator):
        assert comparison({"field": "meta.year", "operator": operator, "value": 2024}) == (
            "year",
            expected_weaviate_operator,
            2024,
        )

    @pytest.mark.parametrize("operator", ORDERING_OPERATORS)
    def test_ordering_operators_normalize_dates(self, operator):
        _, _, value = comparison({"field": "meta.date", "operator": operator, "value": "2024-01-15"})

        assert value == "2024-01-15T00:00:00.000000Z"

    @pytest.mark.parametrize("operator", ORDERING_OPERATORS)
    def test_ordering_operators_against_none_match_no_document(self, operator):
        assert comparison({"field": "meta.year", "operator": operator, "value": None}) == described(
            _match_no_document("year")
        )

    @pytest.mark.parametrize("operator", ORDERING_OPERATORS)
    def test_ordering_operators_reject_strings_that_are_not_dates(self, operator):
        with pytest.raises(FilterError, match="Strings are only comparable if they are ISO formatted dates"):
            _parse_comparison_condition({"field": "meta.kind", "operator": operator, "value": "book"})

    @pytest.mark.parametrize("operator", ORDERING_OPERATORS)
    def test_ordering_operators_reject_lists(self, operator):
        with pytest.raises(FilterError, match="Filter value can't be of type"):
            _parse_comparison_condition({"field": "meta.year", "operator": operator, "value": [2024]})

    def test_in(self):
        assert comparison({"field": "meta.year", "operator": "in", "value": [2020, 2024]}) == described(
            _property("year").contains_any([2020, 2024])
        )

    def test_not_in_requires_every_value_to_differ(self):
        assert comparison({"field": "meta.year", "operator": "not in", "value": [2020, 2024]}) == described(
            Filter.all_of([_property("year").not_equal(2020), _property("year").not_equal(2024)])
        )

    @pytest.mark.parametrize("operator", ["in", "not in"])
    def test_in_and_not_in_require_a_list(self, operator):
        with pytest.raises(FilterError, match="value must be a list when using 'in' or 'not in'"):
            _parse_comparison_condition({"field": "meta.year", "operator": operator, "value": 2024})

    def test_contains_matches_a_substring(self):
        assert comparison({"field": "meta.kind", "operator": "contains", "value": "boo"}) == described(
            _property("kind").like("*boo*")
        )

    def test_contains_requires_a_string(self):
        with pytest.raises(FilterError, match="must be a string when using 'contains'"):
            _parse_comparison_condition({"field": "meta.kind", "operator": "contains", "value": 1})

    def test_the_meta_prefix_is_stripped_because_documents_are_flattened(self):
        assert comparison({"field": "meta.year", "operator": "==", "value": 2024}) == described(
            _property("year").equal(2024)
        )

    def test_a_field_without_the_meta_prefix_is_left_alone(self):
        assert comparison({"field": "content", "operator": "==", "value": "text"}) == described(
            _property("content").equal("text")
        )


class TestMatchNoDocument:
    def test_requires_the_field_to_be_both_set_and_unset(self):
        assert described(_match_no_document("year")) == described(
            Filter.all_of([_property("year").is_none(False), _property("year").is_none(True)])
        )


class TestLogicalOperators:
    def test_and(self):
        condition = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.kind", "operator": "==", "value": "book"},
            ],
        }

        assert described(_parse_logical_condition(condition)) == described(
            Filter.all_of([_property("year").equal(2024), _property("kind").equal("book")])
        )

    def test_or(self):
        condition = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.kind", "operator": "==", "value": "book"},
            ],
        }

        assert described(_parse_logical_condition(condition)) == described(
            Filter.any_of([_property("year").equal(2024), _property("kind").equal("book")])
        )

    def test_nested_logical_conditions(self):
        condition = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.kind", "operator": "==", "value": "book"},
                        {"field": "meta.kind", "operator": "==", "value": "paper"},
                    ],
                },
            ],
        }

        assert described(_parse_logical_condition(condition)) == described(
            Filter.all_of(
                [
                    _property("year").equal(2024),
                    Filter.any_of([_property("kind").equal("book"), _property("kind").equal("paper")]),
                ]
            )
        )

    def test_not_is_pushed_down_into_the_conditions(self):
        # Weaviate has no NOT, so NOT(A AND B) is rewritten as (not A) OR (not B).
        condition = {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.rank", "operator": ">", "value": 1},
            ],
        }

        assert described(_parse_logical_condition(condition)) == described(
            Filter.any_of(
                [
                    _property("year").not_equal(2024) | _property("year").is_none(True),
                    _property("rank").less_or_equal(1),
                ]
            )
        )


class TestConvertFilters:
    def test_a_bare_comparison_is_wrapped(self):
        assert described(convert_filters({"field": "meta.year", "operator": "==", "value": 2024})) == described(
            _property("year").equal(2024)
        )
