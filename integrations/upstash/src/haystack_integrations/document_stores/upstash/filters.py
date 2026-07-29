from typing import Any

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> str:
    """
    Converts Haystack filters into Upstash Vector SQL-like filter strings.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR", "NOT"]:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    parsed_conditions = [_normalize_filters(c) for c in condition["conditions"]]
    if operator == "NOT":
        if len(parsed_conditions) != 1:
            msg = "NOT operator needs exactly one condition"
            raise FilterError(msg)
        return f"NOT ({parsed_conditions[0]})"
    else:
        joined = f" {operator} ".join(f"({c})" for c in parsed_conditions)
        return joined


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
    if "field" not in condition:
        return _parse_logical_condition(condition)

    field = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    if field.startswith("meta."):
        field = field[5:]

    operator = condition["operator"]
    value = condition["value"]

    return COMPARISON_OPERATORS[operator](field, value)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    msg = f"Unsupported value type {type(value)}"
    raise FilterError(msg)


def _equal(field: str, value: Any) -> str:
    return f"{field} = {_format_value(value)}"


def _not_equal(field: str, value: Any) -> str:
    return f"{field} != {_format_value(value)}"


def _greater_than(field: str, value: Any) -> str:
    return f"{field} > {_format_value(value)}"


def _greater_than_equal(field: str, value: Any) -> str:
    return f"{field} >= {_format_value(value)}"


def _less_than(field: str, value: Any) -> str:
    return f"{field} < {_format_value(value)}"


def _less_than_equal(field: str, value: Any) -> str:
    return f"{field} <= {_format_value(value)}"


def _in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = "Value for 'in' must be a list"
        raise FilterError(msg)
    formatted_values = ", ".join(_format_value(v) for v in value)
    return f"{field} IN ({formatted_values})"


def _not_in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = "Value for 'not in' must be a list"
        raise FilterError(msg)
    formatted_values = ", ".join(_format_value(v) for v in value)
    return f"{field} NOT IN ({formatted_values})"


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
