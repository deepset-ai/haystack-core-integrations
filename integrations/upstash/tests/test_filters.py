from haystack_integrations.document_stores.upstash.filters import _normalize_filters


def test_equal_filter():
    f = {"operator": "==", "field": "meta.genre", "value": "fantasy"}
    assert _normalize_filters(f) == "genre = 'fantasy'"


def test_and_filter():
    f = {
        "operator": "AND",
        "conditions": [
            {"operator": "==", "field": "meta.genre", "value": "fantasy"},
            {"operator": ">", "field": "meta.year", "value": 2020},
        ],
    }
    assert _normalize_filters(f) == "(genre = 'fantasy') AND (year > 2020)"


def test_in_filter():
    f = {"operator": "in", "field": "meta.genre", "value": ["fantasy", "action"]}
    assert _normalize_filters(f) == "genre IN ('fantasy', 'action')"


def test_not_filter():
    f = {"operator": "NOT", "conditions": [{"operator": "==", "field": "meta.genre", "value": "fantasy"}]}
    assert _normalize_filters(f) == "NOT (genre = 'fantasy')"


def test_escape_quotes():
    f = {"operator": "==", "field": "meta.author", "value": "O'Brian"}
    assert _normalize_filters(f) == "author = 'O''Brian'"
