# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.document_stores.oracle.filters import FilterTranslator


def _translate(filters):
    params = {}
    counter = [0]
    sql = FilterTranslator().translate(filters, params, counter)
    return sql, params


def test_equality():
    sql, params = _translate({"field": "meta.author", "operator": "==", "value": "Alice"})
    assert "JSON_VALUE(metadata, '$.author') = :p0" in sql
    assert params == {"p0": "Alice"}


def test_inequality():
    sql, params = _translate({"field": "meta.status", "operator": "!=", "value": "draft"})
    assert "!= :p0" in sql
    assert params["p0"] == "draft"


def test_greater_than():
    sql, params = _translate({"field": "meta.year", "operator": ">", "value": 2020})
    assert "TO_NUMBER" in sql
    assert "> :p0" in sql
    assert params["p0"] == 2020


def test_in_operator():
    sql, params = _translate({"field": "meta.lang", "operator": "in", "value": ["en", "de", "fr"]})
    assert "IN (:p0, :p1, :p2)" in sql
    assert params == {"p0": "en", "p1": "de", "p2": "fr"}


def test_not_in_operator():
    sql, _ = _translate({"field": "meta.lang", "operator": "not in", "value": ["xx", "yy"]})
    assert "NOT IN (:p0, :p1)" in sql


def test_and_logical():
    sql, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.author", "operator": "==", "value": "Alice"},
                {"field": "meta.year", "operator": ">", "value": 2020},
            ],
        }
    )
    assert sql.startswith("(")
    assert " AND " in sql
    assert len(params) == 2


def test_or_logical():
    sql, _ = _translate(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": "x"},
                {"field": "meta.b", "operator": "==", "value": "y"},
            ],
        }
    )
    assert " OR " in sql


def test_not_logical():
    sql, _ = _translate(
        {
            "operator": "NOT",
            "conditions": [{"field": "meta.hidden", "operator": "==", "value": True}],
        }
    )
    assert sql.startswith("(NOT ")


def test_nested_and_or():
    sql, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.type", "operator": "==", "value": "article"},
                        {"field": "meta.type", "operator": "==", "value": "blog"},
                    ],
                },
            ],
        }
    )
    assert " AND " in sql
    assert " OR " in sql
    assert len(params) == 3


def test_id_field_maps_to_id_column():
    sql, _ = _translate({"field": "id", "operator": "==", "value": "ABCD1234"})
    assert "id = :p0" in sql


def test_content_field_maps_to_text():
    sql, _ = _translate({"field": "content", "operator": "==", "value": "hello"})
    assert "text = :p0" in sql


def test_numeric_value_wraps_in_to_number():
    sql, _ = _translate({"field": "meta.count", "operator": ">=", "value": 5})
    assert "TO_NUMBER(" in sql
    assert ">= :p0" in sql


def test_nested_meta_key():
    sql, _ = _translate({"field": "meta.author.city", "operator": "==", "value": "NYC"})
    assert "'$.author.city'" in sql


def test_param_counter_increments_correctly():
    _, params = _translate(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": "x"},
                {"field": "meta.b", "operator": "==", "value": "y"},
                {"field": "meta.c", "operator": "==", "value": "z"},
            ],
        }
    )
    assert set(params.keys()) == {"p0", "p1", "p2"}
