# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from dateutil import parser
from haystack.errors import FilterError

import weaviate
from weaviate.collections.classes.filters import Filter, FilterReturn


def validate_filters(filters: dict[str, Any] | None) -> None:
    """
    Validates that filters have the correct structure.

    :param filters: The filters to validate.
    :raises ValueError: If filters are provided but have invalid syntax.
    """
    if filters and "operator" not in filters and "conditions" not in filters:
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise ValueError(msg)


def convert_filters(filters: dict[str, Any]) -> FilterReturn:
    """
    Convert filters from Haystack format to Weaviate format.

    Supported comparison operators: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``,
    ``in``, ``not in``, ``contains``.

    Note: The ``contains`` operator performs substring matching and is
    **case-sensitive**. For case-insensitive matching, normalize the value
    (e.g., lowercase) before building the filter.

    Note: ``NOT`` is translated to Weaviate's native ``NOT`` operator, which requires
    Weaviate 1.33 or later. Older servers reject it at the gRPC layer.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return Filter.all_of([_parse_comparison_condition(filters)])
    return _parse_logical_condition(filters)


LOGICAL_OPERATORS = {
    "AND": Filter.all_of,
    "OR": Filter.any_of,
}


def _parse_logical_condition(condition: dict[str, Any]) -> FilterReturn:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator in ["AND", "OR"]:
        return LOGICAL_OPERATORS[operator](_parse_operands(condition["conditions"]))
    elif operator == "NOT":
        # A NOT node negates the conjunction of its conditions, so wrap them in an AND first.
        return Filter.not_(Filter.all_of(_parse_operands(condition["conditions"])))
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _parse_operands(conditions: list[dict[str, Any]]) -> list[FilterReturn]:
    return [_parse_comparison_condition(c) if "field" in c else _parse_logical_condition(c) for c in conditions]


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
    if isinstance(value, list):
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
    if isinstance(value, list):
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
    if isinstance(value, list):
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
    if isinstance(value, list):
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


def _contains(field: str, value: Any) -> FilterReturn:
    """
    Creates a filter for substring matching using Weaviate's 'like' operator.

    The matching is case-sensitive. For case-insensitive matching, consider
    normalizing the value before passing it to this function.
    """
    if not isinstance(value, str):
        msg = "Filter value must be a string when using 'contains' comparator"
        raise FilterError(msg)
    return weaviate.classes.query.Filter.by_property(field).like(f"*{value}*")


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
    "contains": _contains,
}


def _parse_comparison_condition(condition: dict[str, Any]) -> FilterReturn:
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

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    return COMPARISON_OPERATORS[operator](field, value)


def _match_no_document(field: str) -> FilterReturn:
    """
    Returns a filter that will match no Document.

    This is used to keep the behavior consistent between different Document Stores.
    """

    operands = [weaviate.classes.query.Filter.by_property(field).is_none(val) for val in [False, True]]
    return Filter.all_of(operands)
