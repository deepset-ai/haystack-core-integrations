import pytest
from haystack.preview.errors import FilterError

from elasticsearch_haystack.filters import _normalize_filters, _normalize_ranges

filters_data = [
    (
        {
            "$and": {
                "type": {"$eq": "article"},
                "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
            }
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
            "$or": [
                {"Type": "News Paper", "Date": {"$lt": "2019-01-01"}},
                {"Type": "Blog Post", "Date": {"$gte": "2019-01-01"}},
            ]
        },
        {
            "bool": {
                "should": [
                    {"match": {"Type": "News Paper"}},
                    {"match": {"Type": "Blog Post"}},
                    {"range": {"Date": {"gte": "2019-01-01", "lt": "2019-01-01"}}},
                ]
            }
        },
    ),
    (
        {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
            }
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
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {"genre": ["economy", "politics"], "publisher": "nytimes"},
        },
        {
            "bool": {
                "must": [
                    {"match": {"type": "article"}},
                    {
                        "bool": {
                            "should": [
                                {"terms": {"genre": ["economy", "politics"]}},
                                {"match": {"publisher": "nytimes"}},
                            ]
                        }
                    },
                    {"range": {"date": {"gte": "2015-01-01", "lt": "2021-01-01"}}},
                    {"range": {"rating": {"gte": 3}}},
                ]
            }
        },
    ),
    ({"text": "A Foo Document 1"}, {"match": {"text": "A Foo Document 1"}}),
    (
        {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}},
        {
            "bool": {
                "should": [
                    {"bool": {"should": [{"match": {"$eq": "name_0"}}, {"match": {"$eq": "name_1"}}]}},
                    {"range": {"number": {"lt": 1.0}}},
                ]
            }
        },
    ),
    (
        {"$and": {"number": {"$and": {"$lte": 2, "$gte": 0}}, "name": {"$in": ["name_0", "name_1"]}}},
        {
            "bool": {
                "must": [
                    {"bool": {"must": [{"range": {"number": {"lte": 2, "gte": 0}}}]}},
                    {"terms": {"name": ["name_0", "name_1"]}},
                ]
            }
        },
    ),
    (
        {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]},
        {
            "bool": {
                "must": [
                    {"terms": {"name": ["name_0", "name_1"]}},
                    {"range": {"number": {"lte": 2, "gte": 0}}},
                ]
            }
        },
    ),
    (
        {"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}},
        {"bool": {"must": [{"range": {"number": {"lte": 2, "gte": 0}}}]}},
    ),
]


@pytest.mark.parametrize("filters, expected", filters_data)
def test_normalize_filters(filters, expected):
    result = _normalize_filters(filters)
    assert result == expected


def test_normalize_filters_raises_with_malformed_filters():
    with pytest.raises(FilterError):
        _normalize_filters("not a filter")

    with pytest.raises(FilterError):
        _normalize_filters({"number": {"page": "100"}})

    with pytest.raises(FilterError):
        _normalize_filters({"number": {"page": {"chapter": "intro"}}})


def test_normalize_ranges():
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]
