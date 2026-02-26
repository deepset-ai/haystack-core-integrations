# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.dataclasses import Document
from haystack.errors import FilterError
from haystack.testing.document_store import FilterDocumentsTest

from haystack_integrations.document_stores.opensearch.filters import _normalize_ranges, normalize_filters

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
    result = normalize_filters(filters)
    assert result == expected


def test_normalize_filters_invalid_operator():
    with pytest.raises(FilterError):
        normalize_filters({"operator": "INVALID", "conditions": []})


def test_normalize_filters_malformed():
    # Missing operator
    with pytest.raises(FilterError):
        normalize_filters({"conditions": []})

    # Missing conditions
    with pytest.raises(FilterError):
        normalize_filters({"operator": "AND"})

    # Missing comparison field
    with pytest.raises(FilterError):
        normalize_filters({"operator": "AND", "conditions": [{"operator": "==", "value": "article"}]})

    # Missing comparison operator
    with pytest.raises(FilterError):
        normalize_filters({"operator": "AND", "conditions": [{"field": "meta.type", "operator": "=="}]})

    # Missing comparison value
    with pytest.raises(FilterError):
        normalize_filters({"operator": "AND", "conditions": [{"field": "meta.type", "value": "article"}]})


def test_normalize_ranges():
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]


@pytest.mark.integration
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        The OpenSearchDocumentStore.filter_documents() method returns documents with their score set.
        We don't want to compare the score, so we set it to None before comparing.
        Embeddings are not exactly the same when retrieved from OpenSearch (float round-trip),
        so we compare them approximately and then set both to None for the final equality check.
        """
        assert len(received) == len(expected)
        received = sorted(received, key=lambda x: x.id)
        expected = sorted(expected, key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc.score = None
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc
