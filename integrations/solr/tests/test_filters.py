# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.solr.filters import escape_query_chars, normalize_filters


class TestEscaping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("plain", "plain"),
            ("a+b", "a\\+b"),
            ("(1+1):2", "\\(1\\+1\\)\\:2"),
            ("a b", "a\\ b"),
            ("back\\slash", "back\\\\slash"),
            ('quote"d', 'quote\\"d'),
            ("a&&b", "a\\&\\&b"),
        ],
    )
    def test_escape_query_chars(self, raw, expected):
        assert escape_query_chars(raw) == expected


class TestComparisons:
    def test_equal_picks_the_field_matching_the_value_type(self):
        """The type code comes from the filter value, so "100" and 100 address different fields."""
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": "100"}) == 'meta_s_page:"100"'
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": 100}) == "meta_l_page:100"
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": 1.5}) == "meta_d_page:1.5"
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": True}) == "meta_b_page:true"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(10, "meta_l_n:10"), (-10, "meta_l_n:\\-10"), (2.5, "meta_d_n:2.5"), (-2.5, "meta_d_n:\\-2.5")],
    )
    def test_negative_numbers_are_escaped(self, value, expected):
        """
        An unescaped leading minus is a Lucene syntax error, and inside a group it silently negates.

        `meta_l_n:-10` fails outright, while `meta_l_n:(-10 OR 100)` quietly means "100 but not 10".
        """
        assert normalize_filters({"field": "meta.n", "operator": "==", "value": value}) == expected

    def test_negative_numbers_are_escaped_in_groups(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [-10, 100]})
        assert clause == "(meta_l_n:(\\-10 OR 100))"

    def test_negative_numbers_are_escaped_in_ranges(self):
        clause = normalize_filters({"field": "meta.n", "operator": ">=", "value": -10})
        assert clause == "meta_l_n:[\\-10 TO *]"

    def test_meta_prefix_is_optional(self):
        with_prefix = normalize_filters({"field": "meta.page", "operator": "==", "value": "x"})
        without_prefix = normalize_filters({"field": "page", "operator": "==", "value": "x"})
        assert with_prefix == without_prefix

    def test_top_level_fields_carry_no_type_code(self):
        assert normalize_filters({"field": "id", "operator": "==", "value": "1"}) == 'id:"1"'
        assert normalize_filters({"field": "content", "operator": "==", "value": "hi"}) == 'content:"hi"'

    def test_equal_none_means_the_key_is_absent(self):
        clause = normalize_filters({"field": "meta.number", "operator": "==", "value": None})
        assert clause.startswith("(*:* -(")
        assert "meta_l_number:[* TO *]" in clause
        assert "meta_s_number:[* TO *]" in clause

    def test_not_equal_none_means_the_key_is_present(self):
        clause = normalize_filters({"field": "meta.number", "operator": "!=", "value": None})
        assert not clause.startswith("(*:* -(")
        assert "meta_l_number:[* TO *]" in clause

    def test_not_equal_also_matches_documents_missing_the_field(self):
        """`None != 100` is true in Python, so a Solr negation is the right translation."""
        assert (
            normalize_filters({"field": "meta.number", "operator": "!=", "value": 100}) == "(*:* -(meta_l_number:100))"
        )

    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            (">", "meta_l_number:{100 TO *}"),
            (">=", "meta_l_number:[100 TO *]"),
            ("<", "meta_l_number:[* TO 100}"),
            ("<=", "meta_l_number:[* TO 100]"),
        ],
    )
    def test_ranges(self, operator, expected):
        assert normalize_filters({"field": "meta.number", "operator": operator, "value": 100}) == expected

    def test_iso_dates_compare_lexicographically_on_a_string_field(self):
        clause = normalize_filters({"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"})
        assert clause == 'meta_s_date:{"1972\\-12\\-11T19\\:54\\:58" TO *}'

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
    def test_non_date_strings_are_rejected(self, operator):
        with pytest.raises(FilterError, match="Strings must be ISO formatted dates"):
            normalize_filters({"field": "meta.page", "operator": operator, "value": "100"})

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
    def test_lists_are_rejected_in_ranges(self, operator):
        with pytest.raises(FilterError, match="can't be of type list"):
            normalize_filters({"field": "meta.number", "operator": operator, "value": [1, 2]})

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
    def test_bools_are_rejected_in_ranges(self, operator):
        with pytest.raises(FilterError, match="can't be of type bool"):
            normalize_filters({"field": "meta.flag", "operator": operator, "value": True})

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
    def test_none_matches_nothing_in_ranges(self, operator):
        assert normalize_filters({"field": "meta.number", "operator": operator, "value": None}) == "(-*:*)"

    def test_in_groups_values_by_type(self):
        clause = normalize_filters({"field": "meta.mixed", "operator": "in", "value": [1, 2, "a"]})
        assert "meta_l_mixed:(1 OR 2)" in clause
        assert 'meta_s_mixed:("a")' in clause

    def test_not_in_is_a_negated_in(self):
        clause = normalize_filters({"field": "meta.number", "operator": "not in", "value": [1, 2]})
        assert clause == "(*:* -((meta_l_number:(1 OR 2))))"

    @pytest.mark.parametrize("operator", ["in", "not in"])
    def test_non_list_values_are_rejected(self, operator):
        with pytest.raises(FilterError, match="must have a list of values"):
            normalize_filters({"field": "meta.number", "operator": operator, "value": 1})

    def test_equal_with_a_list_matches_every_element(self):
        """A multi-valued field equals a list when it holds all of its elements."""
        clause = normalize_filters({"field": "meta.tags", "operator": "==", "value": ["a", "b"]})
        assert clause == '(meta_ss_tags:"a" AND meta_ss_tags:"b")'

    def test_equal_with_an_empty_list_means_the_key_is_absent(self):
        clause = normalize_filters({"field": "meta.tags", "operator": "==", "value": []})
        assert clause.startswith("(*:* -(")

    def test_in_with_an_empty_list_matches_nothing(self):
        assert normalize_filters({"field": "meta.n", "operator": "in", "value": []}) == "(-*:*)"

    def test_in_with_only_nulls_means_the_key_is_absent(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [None]})
        assert clause.startswith("(*:* -(")
        assert "meta_l_n:[* TO *]" in clause

    def test_in_mixing_nulls_and_values_covers_both(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [1, None]})
        assert "meta_l_n:(1)" in clause
        assert "-(" in clause

    def test_existence_of_a_top_level_field_needs_no_type_code(self):
        assert normalize_filters({"field": "id", "operator": "!=", "value": None}) == "id:[* TO *]"

    def test_unknown_operator(self):
        with pytest.raises(FilterError, match="Unknown operator"):
            normalize_filters({"field": "meta.number", "operator": "~=", "value": 1})


class TestLogicalConditions:
    def test_and(self):
        clause = normalize_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.page", "operator": "==", "value": "100"},
                    {"field": "meta.number", "operator": ">", "value": 1},
                ],
            }
        )
        assert clause == '(meta_s_page:"100" AND meta_l_number:{1 TO *})'

    def test_or(self):
        clause = normalize_filters(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.page", "operator": "==", "value": "100"},
                    {"field": "meta.page", "operator": "==", "value": "90"},
                ],
            }
        )
        assert clause == '(meta_s_page:"100" OR meta_s_page:"90")'

    def test_not(self):
        clause = normalize_filters(
            {"operator": "NOT", "conditions": [{"field": "meta.page", "operator": "==", "value": "100"}]}
        )
        assert clause == '(*:* -(meta_s_page:"100"))'

    def test_nesting(self):
        clause = normalize_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.number", "operator": "==", "value": 2},
                            {"field": "meta.number", "operator": "==", "value": -10},
                        ],
                    },
                ],
            }
        )
        assert clause == '(meta_s_chapter:"intro" AND (meta_l_number:2 OR meta_l_number:\\-10))'

    def test_unknown_logical_operator(self):
        with pytest.raises(FilterError, match="Unknown logical operator"):
            normalize_filters({"operator": "XOR", "conditions": []})


class TestMalformedFilters:
    def test_non_dict(self):
        with pytest.raises(FilterError, match="Filters must be a dictionary"):
            normalize_filters(["not", "a", "dict"])

    def test_missing_operator_key(self):
        with pytest.raises(FilterError, match="'operator' key missing"):
            normalize_filters({"conditions": []})

    def test_missing_conditions_key(self):
        with pytest.raises(FilterError, match="'conditions' key missing"):
            normalize_filters({"operator": "AND"})

    def test_missing_value_key(self):
        with pytest.raises(FilterError, match="'value' key missing"):
            normalize_filters({"field": "meta.page", "operator": "=="})

    def test_missing_condition_operator_key(self):
        with pytest.raises(FilterError, match="'operator' key missing"):
            normalize_filters({"field": "meta.page", "value": "100"})

    def test_explicit_none_field(self):
        with pytest.raises(FilterError, match="'field' key missing"):
            normalize_filters({"field": None, "operator": "==", "value": "100"})

    def test_empty_conditions_match_everything(self):
        assert normalize_filters({"operator": "AND", "conditions": []}) == "*:*"
