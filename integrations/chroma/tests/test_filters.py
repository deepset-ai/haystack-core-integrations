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


@pytest.mark.parametrize(
    ("operator", "value", "chroma_operator"),
    [
        ("==", "target", "$eq"),
        ("!=", "target", "$ne"),
        ("in", ["target"], "$in"),
        ("not in", ["target"], "$nin"),
        ("==", "", "$eq"),
    ],
)
def test_metadata_id_filters_keep_metadata_comparisons(operator, value, chroma_operator):
    result = _convert_filters({"field": "meta.id", "operator": operator, "value": value})

    assert result.ids == []
    assert result.where == {"id": {chroma_operator: value}}
    assert result.where_document is None


def test_document_id_filter_uses_ids():
    result = _convert_filters({"field": "id", "operator": "==", "value": "target"})

    assert result.ids == ["target"]
    assert result.where is None
    assert result.where_document is None


@pytest.mark.parametrize(
    ("condition", "category"),
    [
        ({"field": "meta.id", "operator": "in", "value": "target"}, "metadata filter"),
        ({"field": "content", "operator": "contains", "value": 123}, "document content filter"),
    ],
    ids=["invalid_metadata_membership", "invalid_content_value"],
)
def test_invalid_filters_wrap_chroma_validation_errors(condition, category):
    with pytest.raises(ChromaDocumentStoreFilterError, match=f"Invalid '{category}'") as exception_info:
        _convert_filters(condition)

    assert isinstance(exception_info.value.__cause__, ValueError)
