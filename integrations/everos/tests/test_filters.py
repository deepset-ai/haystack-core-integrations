# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.memory_stores.everos.filters import build_search_filters


def test_build_search_filters_converts_comparison_and_session_scope():
    result = build_search_filters(
        filters={"field": "timestamp", "operator": ">=", "value": 1_700_000_000_000},
        session_id="session-1",
    )
    assert result == {
        "AND": [
            {"timestamp": {"gte": 1_700_000_000_000}},
            {"session_id": "session-1"},
        ]
    }


def test_build_search_filters_converts_logical_tree():
    result = build_search_filters(
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "session_id", "operator": "==", "value": "one"},
                {"field": "session_id", "operator": "in", "value": ["two", "three"]},
            ],
        }
    )
    assert result == {"OR": [{"session_id": "one"}, {"session_id": {"in": ["two", "three"]}}]}


def test_build_search_filters_rejects_unsupported_operator():
    with pytest.raises(ValueError, match="Unsupported EverOS filter operator"):
        build_search_filters(filters={"field": "session_id", "operator": "not in", "value": ["one"]})


def test_build_search_filters_returns_session_filter_directly():
    assert build_search_filters(session_id="session-1") == {"session_id": "session-1"}


@pytest.mark.parametrize(
    ("filters", "exception", "match"),
    [
        ({"field": "", "operator": "==", "value": "x"}, ValueError, "non-empty string"),
        (
            {"operator": "NOT", "conditions": [{"field": "session_id", "operator": "==", "value": "x"}]},
            ValueError,
            "logical",
        ),
        ({"operator": "AND", "conditions": []}, ValueError, "at least one"),
        ({"operator": "AND", "conditions": ["invalid"]}, TypeError, "dictionary"),
        ({"session_id": "native-shape"}, ValueError, "Haystack comparison"),
    ],
)
def test_build_search_filters_rejects_invalid_shapes(filters, exception, match):
    with pytest.raises(exception, match=match):
        build_search_filters(filters=filters)
