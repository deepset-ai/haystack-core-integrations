# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.chroma.errors import ChromaDocumentStoreFilterError
from haystack_integrations.document_stores.chroma.filters import (
    _convert_filters,
    _parse_comparison_condition,
    _parse_logical_condition,
)


def test_id_filter_with_empty_value_raises():
    with pytest.raises(ChromaDocumentStoreFilterError, match="id filter only supports"):
        _convert_filters({"field": "id", "operator": "==", "value": ""})


def test_id_filter_with_non_eq_operator_raises():
    # A non-'==' operator produces e.g. {"$in": [...]} (no "$eq" key), which must
    # surface the friendly ChromaDocumentStoreFilterError, not a raw KeyError('$eq').
    with pytest.raises(ChromaDocumentStoreFilterError, match="id filter only supports"):
        _convert_filters({"field": "id", "operator": "in", "value": ["1", "2"]})


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        ({"conditions": []}, "'operator' key missing"),
        ({"operator": "AND"}, "'conditions' key missing"),
        ({"operator": "XOR", "conditions": []}, "Unknown operator"),
    ],
    ids=["missing_operator", "missing_conditions", "unknown_operator"],
)
def test_parse_logical_condition_errors(condition, match):
    with pytest.raises(ChromaDocumentStoreFilterError, match=match):
        _parse_logical_condition(condition)


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        ({"operator": "==", "value": "x"}, "'field' key missing"),
        ({"field": "meta.a", "value": "x"}, "'operator' key missing"),
        ({"field": "meta.a", "operator": "=="}, "'value' key missing"),
        ({"field": "meta.a", "operator": "~~", "value": "x"}, "Unknown operator"),
    ],
    ids=["missing_field", "missing_operator", "missing_value", "unknown_operator"],
)
def test_parse_comparison_condition_errors(condition, match):
    with pytest.raises(ChromaDocumentStoreFilterError, match=match):
        _parse_comparison_condition(condition)


@pytest.mark.parametrize("content_first", [True, False])
def test_and_filter_mixing_content_and_metadata(content_first):
    content = {"field": "content", "operator": "contains", "value": "x"}
    meta = {"field": "meta.a", "operator": "==", "value": 1}
    conditions = [content, meta] if content_first else [meta, content]
    result = _convert_filters({"operator": "AND", "conditions": conditions})
    assert result.ids == []
    assert result.where == {"a": {"$eq": 1}}
    assert result.where_document == {"$contains": "x"}


def test_and_filter_mixing_content_and_several_metadata_conditions():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.a", "operator": "==", "value": 1},
            {"field": "content", "operator": "contains", "value": "x"},
            {"field": "meta.b", "operator": ">", "value": 2},
        ],
    }
    result = _convert_filters(filters)
    assert result.where == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}
    assert result.where_document == {"$contains": "x"}
