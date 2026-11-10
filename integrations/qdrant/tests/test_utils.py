# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.document_stores.types import FilterPolicy
from qdrant_client.http import models as rest

from haystack_integrations.components.retrievers.qdrant.utils import _resolve_filters


@pytest.mark.parametrize(
    ("init_filters", "runtime_filters"),
    [(rest.Filter(must=[]), None), (None, rest.Filter(must=[]))],
    ids=["init", "runtime"],
)
def test_merge_rejects_native_filters(init_filters, runtime_filters):
    with pytest.raises(ValueError, match="Native Qdrant filters"):
        _resolve_filters(filter_policy=FilterPolicy.MERGE, init_filters=init_filters, runtime_filters=runtime_filters)


def test_replace_with_native_filters():
    native_filter = rest.Filter(must=[rest.FieldCondition(key="meta.kind", match=rest.MatchValue(value="A"))])
    haystack_filter = {"field": "meta.kind", "operator": "==", "value": "B"}

    assert (
        _resolve_filters(
            filter_policy=FilterPolicy.REPLACE, init_filters=native_filter, runtime_filters=haystack_filter
        )
        == haystack_filter
    )
    assert (
        _resolve_filters(
            filter_policy=FilterPolicy.REPLACE, init_filters=haystack_filter, runtime_filters=native_filter
        )
        == native_filter
    )
    assert (
        _resolve_filters(filter_policy=FilterPolicy.REPLACE, init_filters=native_filter, runtime_filters=None)
        == native_filter
    )


def test_merge_haystack_filters():
    init_filters = {"field": "meta.kind", "operator": "==", "value": "A"}
    runtime_filters = {"field": "meta.page", "operator": "==", "value": 2}

    assert _resolve_filters(
        filter_policy=FilterPolicy.MERGE, init_filters=init_filters, runtime_filters=runtime_filters
    ) == {"operator": "AND", "conditions": [init_filters, runtime_filters]}
