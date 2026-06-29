# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.mongodb_atlas.filters import _normalize_filters


class TestNormalizeFilters:
    @pytest.mark.parametrize(
        ("filters", "expected_error", "match"),
        [
            ("not a dict", FilterError, "must be a dictionary"),
            ({"field": "meta.x"}, ValueError, "Invalid filter syntax"),
            ({"conditions": []}, FilterError, "'operator' key missing"),
            ({"operator": "AND"}, FilterError, "'conditions' key missing"),
            ({"operator": "XOR", "conditions": []}, FilterError, "Unknown logical operator"),
            ({"field": "meta.x", "operator": ">", "value": [1, 2]}, FilterError, "Cant compare"),
            ({"field": "meta.x", "operator": "<", "value": "not-a-date"}, FilterError, "ISO formatted dates"),
            ({"field": "meta.x", "operator": "in", "value": "single"}, FilterError, "must be a list"),
            ({"field": "meta.x", "operator": "not in", "value": "single"}, FilterError, "must be a list"),
        ],
    )
    def test_invalid_filters_raise(self, filters, expected_error, match):
        with pytest.raises(expected_error, match=match):
            _normalize_filters(filters)

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (
                {"operator": "NOT", "conditions": [{"field": "meta.x", "operator": "==", "value": 1}]},
                {"$nor": [{"$and": [{"meta.x": {"$eq": 1}}]}]},
            ),
            ({"field": "meta.x", "operator": ">=", "value": None}, {"meta.x": {"$gt": None}}),
            ({"field": "meta.x", "operator": "<=", "value": None}, {"meta.x": {"$lt": None}}),
        ],
    )
    def test_valid_filters_are_normalized(self, filters, expected):
        assert _normalize_filters(filters) == expected
