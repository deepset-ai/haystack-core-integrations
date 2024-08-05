# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from dateutil import parser
from haystack.errors import FilterError
from pandas import DataFrame

import weaviate
from weaviate.collections.classes.filters import Filter, FilterReturn


def convert_filters(filters: Dict[str, Any]) -> FilterReturn:
    """
    Convert filters from Haystack format to Weaviate format.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return Filter.all_of([_parse_comparison_condition(filters)])
    return _parse_logical_condition(filters)


OPERATOR_INVERSE = {
    "==": "!=",
    "!=": "==",
    ">": "<=",
    ">=": "<",
    "<": ">=",
    "<=": ">",
    "in": "not in",
    "not in": "in",
    "AND": "OR",
    "OR": "AND",
    "NOT": "OR",
}


def _invert_condition(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invert condition recursively.
    Weaviate doesn't support NOT filters so we need to invert them ourselves.
    """
    inverted_condition = filters.copy()
    if "operator" not in filters:
        # Let's not handle this stuff in here, we'll fail later on anyway.
        return inverted_condition
    inverted_condition["operator"] = OPERATOR_INVERSE[filters["operator"]]
    if "conditions" in filters:
        inverted_condition["conditions"] = []
        for condition in filters["conditions"]:
            inverted_condition["conditions"].append(_invert_condition(condition))

    return inverted_condition


LOGICAL_OPERATORS = {
    "AND": Filter.all_of,
    "OR": Filter.any_of,
}


def _parse_logical_condition(condition: Dict[str, Any]) -> FilterReturn:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator in ["AND", "OR"]:
        operands = []
        for c in condition["conditions"]:
            if "field" not in c:
                operands.append(_parse_logical_condition(c))
            else:
                operands.append(_parse_comparison_condition(c))
        return LOGICAL_OPERATORS[operator](operands)
    elif operator == "NOT":
        inverted_conditions = _invert_condition(condition)
        return _parse_logical_condition(inverted_conditions)
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _handle_date(value: Any) -> str:
    if isinstance(value, str):
        try:
            return parser.isoparse(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            pass
    return value


def _equal(field: str, value: Any) -> FilterReturn:
    if value is None:
        return weaviate.classes.query.Filter.by_property(field).is_none(True)
    return weaviate.classes.query.Filter.by_property(field).equal(_handle_date(value))


def _not_equal(field: str, value: Any) -> FilterReturn:
    if value is None:
        return weaviate.classes.query.Filter.by_property(field).is_none(False)

    return weaviate.classes.query.Filter.by_property(field).not_equal(
        _handle_date(value)
    ) | weaviate.classes.query.Filter.by_property(field).is_none(True)


def _greater_than(field: str, value: Any) -> FilterReturn:
    if value is None:
        # When the value is None and '>' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return weaviate.classes.query.Filter.by_property(field).greater_than(_handle_date(value))


def _greater_than_equal(field: str, value: Any) -> FilterReturn:
    if value is None:
        # When the value is None and '>=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return weaviate.classes.query.Filter.by_property(field).greater_or_equal(_handle_date(value))


def _less_than(field: str, value: Any) -> FilterReturn:
    if value is None:
        # When the value is None and '<' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return weaviate.classes.query.Filter.by_property(field).less_than(_handle_date(value))


def _less_than_equal(field: str, value: Any) -> FilterReturn:
    if value is None:
        # When the value is None and '<=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return weaviate.classes.query.Filter.by_property(field).less_or_equal(_handle_date(value))


def _in(field: str, value: Any) -> FilterReturn:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)

    return weaviate.classes.query.Filter.by_property(field).contains_any(value)


def _not_in(field: str, value: Any) -> FilterReturn:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    operands = [weaviate.classes.query.Filter.by_property(field).not_equal(v) for v in value]
    return Filter.all_of(operands)


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


def _parse_comparison_condition(condition: Dict[str, Any]) -> FilterReturn:
    field: str = condition["field"]

    if field.startswith("meta."):
        # Documents are flattened otherwise we wouldn't be able to properly query them.
        # We're forced to flatten because Weaviate doesn't support querying of nested properties
        # as of now. If we don't flatten the documents we can't filter them.
        # As time of writing this they have it in their backlog, see:
        # https://github.com/weaviate/weaviate/issues/3694
        field = field.replace("meta.", "")

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


def _match_no_document(field: str) -> FilterReturn:
    """
    Returns a filters that will match no Document, this is used to keep the behavior consistent
    between different Document Stores.
    """

    operands = [weaviate.classes.query.Filter.by_property(field).is_none(val) for val in [False, True]]
    return Filter.all_of(operands)
