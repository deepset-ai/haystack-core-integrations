# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.amazon_s3_vectors.filters import _normalize_filters, _validate_filters


class TestNormalizeFilters:
    def test_simple_equality(self):
        filters = {"field": "meta.category", "operator": "==", "value": "news"}
        result = _normalize_filters(filters)
        assert result == {"category": {"$eq": "news"}}

    def test_not_equal(self):
        filters = {"field": "meta.status", "operator": "!=", "value": "draft"}
        result = _normalize_filters(filters)
        assert result == {"status": {"$ne": "draft"}}

    def test_greater_than(self):
        filters = {"field": "meta.year", "operator": ">", "value": 2020}
        result = _normalize_filters(filters)
        assert result == {"year": {"$gt": 2020}}

    def test_greater_than_equal(self):
        filters = {"field": "meta.year", "operator": ">=", "value": 2020}
        result = _normalize_filters(filters)
        assert result == {"year": {"$gte": 2020}}

    def test_less_than(self):
        filters = {"field": "meta.price", "operator": "<", "value": 50.0}
        result = _normalize_filters(filters)
        assert result == {"price": {"$lt": 50.0}}

    def test_less_than_equal(self):
        filters = {"field": "meta.price", "operator": "<=", "value": 50.0}
        result = _normalize_filters(filters)
        assert result == {"price": {"$lte": 50.0}}

    def test_in_operator(self):
        filters = {"field": "meta.genre", "operator": "in", "value": ["comedy", "drama"]}
        result = _normalize_filters(filters)
        assert result == {"genre": {"$in": ["comedy", "drama"]}}

    def test_not_in_operator(self):
        filters = {"field": "meta.genre", "operator": "not in", "value": ["horror"]}
        result = _normalize_filters(filters)
        assert result == {"genre": {"$nin": ["horror"]}}

    def test_and_logical(self):
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.genre", "operator": "==", "value": "drama"},
                {"field": "meta.year", "operator": ">=", "value": 2020},
            ],
        }
        result = _normalize_filters(filters)
        assert result == {"$and": [{"genre": {"$eq": "drama"}}, {"year": {"$gte": 2020}}]}

    def test_or_logical(self):
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.genre", "operator": "==", "value": "drama"},
                {"field": "meta.genre", "operator": "==", "value": "comedy"},
            ],
        }
        result = _normalize_filters(filters)
        assert result == {"$or": [{"genre": {"$eq": "drama"}}, {"genre": {"$eq": "comedy"}}]}

    def test_field_without_meta_prefix(self):
        filters = {"field": "category", "operator": "==", "value": "news"}
        result = _normalize_filters(filters)
        assert result == {"category": {"$eq": "news"}}

    def test_unsupported_type_raises(self):
        filters = {"field": "meta.data", "operator": "==", "value": [1, 2, 3]}
        with pytest.raises(FilterError):
            _normalize_filters(filters)

    def test_in_with_non_list_raises(self):
        filters = {"field": "meta.genre", "operator": "in", "value": "comedy"}
        with pytest.raises(FilterError):
            _normalize_filters(filters)

    def test_missing_operator_raises(self):
        filters = {"field": "meta.genre", "value": "comedy"}
        with pytest.raises(FilterError):
            _normalize_filters(filters)

    def test_missing_value_raises(self):
        filters = {"field": "meta.genre", "operator": "=="}
        with pytest.raises(FilterError):
            _normalize_filters(filters)

    def test_invalid_dict_raises(self):
        with pytest.raises(FilterError):
            _normalize_filters("not a dict")


class TestValidateFilters:
    def test_valid_filters(self):
        _validate_filters({"operator": "AND", "conditions": []})

    def test_none_is_valid(self):
        _validate_filters(None)

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            _validate_filters({"field": "meta.x"})
