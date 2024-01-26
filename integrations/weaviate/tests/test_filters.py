from haystack_integrations.document_stores.weaviate._filters import _invert_condition


def test_invert_conditions():
    filter = {
        "operator": "NOT",
        "conditions": [
            {"field": "meta.number", "operator": "==", "value": 100},
            {"field": "meta.name", "operator": "==", "value": "name_0"},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "name_1"},
                    {"field": "meta.name", "operator": "==", "value": "name_2"},
                ],
            },
        ],
    }

    inverted = _invert_condition(filter)
    assert inverted == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.number", "operator": "!=", "value": 100},
            {"field": "meta.name", "operator": "!=", "value": "name_0"},
            {
                "conditions": [
                    {"field": "meta.name", "operator": "!=", "value": "name_1"},
                    {"field": "meta.name", "operator": "!=", "value": "name_2"},
                ],
                "operator": "AND",
            },
        ],
    }
