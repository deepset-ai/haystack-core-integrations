from datetime import datetime
from typing import Any, Dict

from haystack.errors import FilterError
from pandas import DataFrame


def convert_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert filters from Haystack format to Weaviate format.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return {"operator": "And", "operands": [_parse_comparison_condition(filters)]}
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
    "NOT": "AND",
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


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator in ["AND", "OR"]:
        return {
            "operator": operator.lower().capitalize(),
            "operands": [_parse_comparison_condition(c) for c in condition["conditions"]],
        }
    elif operator == "NOT":
        inverted_conditions = _invert_condition(condition)
        return _parse_logical_condition(inverted_conditions)
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _infer_value_type(value: Any) -> str:
    if value is None:
        return "valueNull"

    if isinstance(value, bool):
        return "valueBoolean"
    if isinstance(value, int):
        return "valueInt"
    if isinstance(value, float):
        return "valueNumber"

    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
            return "valueDate"
        except ValueError:
            return "valueText"

    msg = f"Unknown value type {type(value)}"
    raise FilterError(msg)


def _handle_date(value: Any) -> str:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            pass
    return value


def _equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {"path": field, "operator": "IsNull", "valueBoolean": True}
    return {"path": field, "operator": "Equal", _infer_value_type(value): _handle_date(value)}


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {"path": field, "operator": "IsNull", "valueBoolean": False}
    return {
        "operator": "Or",
        "operands": [
            {"path": field, "operator": "NotEqual", _infer_value_type(value): _handle_date(value)},
            {"path": field, "operator": "IsNull", "valueBoolean": True},
        ],
    }


def _greater_than(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '>' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {
            "operator": "And",
            "operands": [
                {"path": field, "operator": "IsNull", "valueBoolean": False},
                {"path": field, "operator": "IsNull", "valueBoolean": True},
            ],
        }
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
    return {"path": field, "operator": "GreaterThan", _infer_value_type(value): _handle_date(value)}


def _greater_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '>=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {
            "operator": "And",
            "operands": [
                {"path": field, "operator": "IsNull", "valueBoolean": False},
                {"path": field, "operator": "IsNull", "valueBoolean": True},
            ],
        }
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
    return {"path": field, "operator": "GreaterThanEqual", _infer_value_type(value): _handle_date(value)}


def _less_than(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '<' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {
            "operator": "And",
            "operands": [
                {"path": field, "operator": "IsNull", "valueBoolean": False},
                {"path": field, "operator": "IsNull", "valueBoolean": True},
            ],
        }
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
    return {"path": field, "operator": "LessThan", _infer_value_type(value): _handle_date(value)}


def _less_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # When the value is None and '<=' is used we create a filter that would return a Document
        # if it has a field set and not set at the same time.
        # This will cause the filter to match no Document.
        # This way we keep the behavior consistent with other Document Stores.
        return {
            "operator": "And",
            "operands": [
                {"path": field, "operator": "IsNull", "valueBoolean": False},
                {"path": field, "operator": "IsNull", "valueBoolean": True},
            ],
        }
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
    return {"path": field, "operator": "LessThanEqual", _infer_value_type(value): _handle_date(value)}


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)

    return {"operator": "And", "operands": [_equal(field, v) for v in value]}


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    return {"operator": "And", "operands": [_not_equal(field, v) for v in value]}


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
