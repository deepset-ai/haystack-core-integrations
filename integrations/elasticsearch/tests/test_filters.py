# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.elasticsearch.filters import _normalize_filters, _normalize_ranges

filters_data = [
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
                {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
                {"field": "meta.date", "operator": "<", "value": "2021-01-01"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
            ],
        },
        {
            "bool": {
                "must": [
                    {"term": {"type": "article"}},
                    {
                        "bool": {
                            "should": [
                                {"terms": {"genre": ["economy", "politics"]}},
                                {"term": {"publisher": "nytimes"}},
                            ]
                        }
                    },
                    {"range": {"date": {"gte": "2015-01-01", "lt": "2021-01-01"}}},
                    {"range": {"rating": {"gte": 3}}},
                ]
            }
        },
    ),
    (
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.Type", "operator": "==", "value": "News Paper"},
                        {"field": "meta.Date", "operator": "<", "value": "2020-01-01"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.Type", "operator": "==", "value": "Blog Post"},
                        {"field": "meta.Date", "operator": ">=", "value": "2019-01-01"},
                    ],
                },
            ],
        },
        {
            "bool": {
                "should": [
                    {"bool": {"must": [{"term": {"Type": "News Paper"}}, {"range": {"Date": {"lt": "2020-01-01"}}}]}},
                    {"bool": {"must": [{"term": {"Type": "Blog Post"}}, {"range": {"Date": {"gte": "2019-01-01"}}}]}},
                ]
            }
        },
    ),
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
                {"field": "meta.date", "operator": "<", "value": "2021-01-01"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        },
        {
            "bool": {
                "must": [
                    {"term": {"type": "article"}},
                    {
                        "bool": {
                            "should": [
                                {"terms": {"genre": ["economy", "politics"]}},
                                {"term": {"publisher": "nytimes"}},
                            ]
                        }
                    },
                    {"range": {"date": {"gte": "2015-01-01", "lt": "2021-01-01"}}},
                    {"range": {"rating": {"gte": 3}}},
                ]
            }
        },
    ),
    (
        {"operator": "AND", "conditions": [{"field": "text", "operator": "==", "value": "A Foo Document 1"}]},
        {"bool": {"must": [{"match": {"text": {"query": "A Foo Document 1", "minimum_should_match": "100%"}}}]}},
    ),
    (
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.name", "operator": "==", "value": "name_0"},
                        {"field": "meta.name", "operator": "==", "value": "name_1"},
                    ],
                },
                {"field": "meta.number", "operator": "<", "value": 1.0},
            ],
        },
        {
            "bool": {
                "should": [
                    {"bool": {"should": [{"term": {"name": "name_0"}}, {"term": {"name": "name_1"}}]}},
                    {"range": {"number": {"lt": 1.0}}},
                ]
            }
        },
    ),
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.number", "operator": "<=", "value": 2},
                {"field": "meta.number", "operator": ">=", "value": 0},
                {"field": "meta.name", "operator": "in", "value": ["name_0", "name_1"]},
            ],
        },
        {"bool": {"must": [{"terms": {"name": ["name_0", "name_1"]}}, {"range": {"number": {"lte": 2, "gte": 0}}}]}},
    ),
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.number", "operator": "<=", "value": 2},
                {"field": "meta.number", "operator": ">=", "value": 0},
            ],
        },
        {"bool": {"must": [{"range": {"number": {"lte": 2, "gte": 0}}}]}},
    ),
    (
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.name", "operator": "==", "value": "name_0"},
                {"field": "meta.name", "operator": "==", "value": "name_1"},
            ],
        },
        {"bool": {"should": [{"term": {"name": "name_0"}}, {"term": {"name": "name_1"}}]}},
    ),
    (
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.number", "operator": "==", "value": 100},
                {"field": "meta.name", "operator": "==", "value": "name_0"},
            ],
        },
        {"bool": {"must_not": [{"bool": {"must": [{"term": {"number": 100}}, {"term": {"name": "name_0"}}]}}]}},
    ),
]


@pytest.mark.parametrize("filters, expected", filters_data)
def test_normalize_filters(filters, expected):
    result = _normalize_filters(filters)
    assert result == expected


def test_normalize_filters_invalid_operator():
    with pytest.raises(FilterError):
        _normalize_filters({"operator": "INVALID", "conditions": []})


def test_normalize_filters_malformed():
    # Missing operator
    with pytest.raises(FilterError):
        _normalize_filters({"conditions": []})

    # Missing conditions
    with pytest.raises(FilterError):
        _normalize_filters({"operator": "AND"})

    # Missing comparison field
    with pytest.raises(FilterError):
        _normalize_filters({"operator": "AND", "conditions": [{"operator": "==", "value": "article"}]})

    # Missing comparison operator
    with pytest.raises(FilterError):
        _normalize_filters({"operator": "AND", "conditions": [{"field": "meta.type", "operator": "=="}]})

    # Missing comparison value
    with pytest.raises(FilterError):
        _normalize_filters({"operator": "AND", "conditions": [{"field": "meta.type", "value": "article"}]})


def test_normalize_ranges():
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]


def test_normalize_filters_rejects_non_dict():
    with pytest.raises(FilterError, match="Filters must be a dictionary"):
        _normalize_filters("not-a-dict")  # type: ignore[arg-type]


def test_equal_none_list_and_text_branches():
    assert _normalize_filters({"field": "meta.k", "operator": "==", "value": None}) == {
        "bool": {"must": {"bool": {"must_not": {"exists": {"field": "k"}}}}}
    }
    out = _normalize_filters({"field": "meta.tags", "operator": "==", "value": ["a", "b"]})
    inner = out["bool"]["must"]
    assert "terms_set" in inner
    assert inner["terms_set"]["tags"]["terms"] == ["a", "b"]


def test_not_equal_none_list_text_and_term():
    ne_none = _normalize_filters({"field": "meta.k", "operator": "!=", "value": None})
    assert ne_none == {"bool": {"must": {"exists": {"field": "k"}}}}
    ne_list = _normalize_filters({"field": "meta.k", "operator": "!=", "value": [1, 2]})
    assert ne_list == {"bool": {"must": {"bool": {"must_not": {"terms": {"k": [1, 2]}}}}}}

    ne_text = _normalize_filters({"field": "text", "operator": "!=", "value": "hello"})
    assert "must_not" in ne_text["bool"]["must"]["bool"]

    ne_term = _normalize_filters({"field": "meta.k", "operator": "!=", "value": "v"})
    assert ne_term == {"bool": {"must": {"bool": {"must_not": {"term": {"k": "v"}}}}}}


@pytest.mark.parametrize(
    ("op", "bound"),
    [
        (">", "gt"),
        (">=", "gte"),
        ("<", "lt"),
        ("<=", "lte"),
    ],
)
def test_range_operators_numeric_and_iso_string(op: str, bound: str):
    body = _normalize_filters({"field": "meta.n", "operator": op, "value": 5})
    assert body == {"bool": {"must": {"range": {"n": {bound: 5}}}}}

    body_dt = _normalize_filters({"field": "meta.d", "operator": op, "value": "2020-01-01"})
    assert body_dt["bool"]["must"]["range"]["d"][bound] == "2020-01-01"


@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_range_operators_none_yields_empty_match(op: str):
    body = _normalize_filters({"field": "meta.x", "operator": op, "value": None})
    inner = body["bool"]["must"]
    assert "bool" in inner and "must" in inner["bool"]
    assert len(inner["bool"]["must"]) == 2


@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_range_operators_reject_non_iso_string(op: str):
    with pytest.raises(FilterError, match="ISO formatted dates"):
        _normalize_filters({"field": "meta.x", "operator": op, "value": "not-a-date"})


@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_range_operators_reject_list_value(op: str):
    with pytest.raises(FilterError, match="Filter value can't be of type"):
        _normalize_filters({"field": "meta.x", "operator": op, "value": [1, 2]})


def test_in_and_not_in_require_list():
    with pytest.raises(FilterError, match="must be a list"):
        _normalize_filters({"field": "meta.x", "operator": "in", "value": "single"})
    with pytest.raises(FilterError, match="must be a list"):
        _normalize_filters({"field": "meta.x", "operator": "not in", "value": 1})


def test_not_in_with_list_value():
    body = _normalize_filters({"field": "meta.x", "operator": "not in", "value": ["a", "b"]})
    assert body == {"bool": {"must": {"bool": {"must_not": {"terms": {"x": ["a", "b"]}}}}}}
