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
