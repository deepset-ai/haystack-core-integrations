# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Any, Dict, List

from haystack.errors import FilterError
from pandas import DataFrame


def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters in OpenSearch compatible filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return {"bool": {"must": _parse_comparison_condition(filters)}}
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]
    if len(conditions) > 1:
        conditions = _normalize_ranges(conditions)
    if operator == "AND":
        return {"bool": {"must": conditions}}
    elif operator == "OR":
        return {"bool": {"should": conditions}}
    elif operator == "NOT":
        return {"bool": {"must_not": [{"bool": {"must": conditions}}]}}
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {"bool": {"must_not": {"exists": {"field": field}}}}

    if isinstance(value, list):
        return {
            "terms_set": {
                field: {
                    "terms": value,
                    "minimum_should_match_script": {"source": f"Math.max(params.num_terms, doc['{field}'].size())"},
                }
            }
        }
    if field in ["text", "dataframe"]:
        # We want to fully match the text field.
        return {"match": {field: {"query": value, "minimum_should_match": "100%"}}}
    return {"term": {field: value}}


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {"exists": {"field": field}}

    if isinstance(value, list):
        return {"bool": {"must_not": {"terms": {field: value}}}}
    if field in ["text", "dataframe"]:
        # We want to fully match the text field.
        return {"bool": {"must_not": {"match": {field: {"query": value, "minimum_should_match": "100%"}}}}}

    return {"bool": {"must_not": {"term": {field: value}}}}


def _greater_than(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '>' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {"bool": {"must": [{"exists": {"field": field}}, {"bool": {"must_not": {"exists": {"field": field}}}}]}}
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return {"range": {field: {"gt": value}}}


def _greater_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '>=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {"bool": {"must": [{"exists": {"field": field}}, {"bool": {"must_not": {"exists": {"field": field}}}}]}}
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return {"range": {field: {"gte": value}}}


def _less_than(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '<' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {"bool": {"must": [{"exists": {"field": field}}, {"bool": {"must_not": {"exists": {"field": field}}}}]}}
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return {"range": {field: {"lt": value}}}


def _less_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '<=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {"bool": {"must": [{"exists": {"field": field}}, {"bool": {"must_not": {"exists": {"field": field}}}}]}}
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return {"range": {field: {"lte": value}}}


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    return {"terms": {field: value}}


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    return {"bool": {"must_not": {"terms": {field: value}}}}


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _parse_logical_condition(condition)
    field: str = condition["field"]

    if field.startswith("meta."):
        # Remove the "meta." prefix if present.
        # Documents are flattened when using the OpenSearchDocumentStore
        # so we don't need to specify the "meta." prefix.
        # Instead of raising an error we handle it gracefully.
        field = field[5:]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]
    if isinstance(value, DataFrame):
        value = value.to_json()

    return COMPARISON_OPERATORS[operator](field, value)


def _normalize_ranges(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges range conditions acting on a same field.

    Example usage:

    ```python
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]
    ```
    """
    range_conditions = [next(iter(c["range"].items())) for c in conditions if "range" in c]
    if range_conditions:
        conditions = [c for c in conditions if "range" not in c]
        range_conditions_dict: Dict[str, Any] = {}
        for field_name, comparison in range_conditions:
            if field_name not in range_conditions_dict:
                range_conditions_dict[field_name] = {}
            range_conditions_dict[field_name].update(comparison)

        for field_name, comparisons in range_conditions_dict.items():
            conditions.append({"range": {field_name: comparisons}})
    return conditions
