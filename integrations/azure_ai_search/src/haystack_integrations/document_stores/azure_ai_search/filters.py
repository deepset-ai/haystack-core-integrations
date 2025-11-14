from typing import Any

from dateutil import parser

from .errors import AzureAISearchDocumentStoreFilterError

LOGICAL_OPERATORS = {"AND": "and", "OR": "or", "NOT": "not"}


def _normalize_filters(filters: dict[str, Any]) -> str:
    """
    Converts Haystack filters in Azure AI Search compatible filters.
    """
    if not isinstance(filters, dict):
        msg = """Filters must be a dictionary.
        See https://docs.haystack.deepset.ai/docs/metadata-filtering for details on filters syntax."""
        raise AzureAISearchDocumentStoreFilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    missing_keys = [key for key in ("operator", "conditions") if key not in condition]
    if missing_keys:
        msg = f"""Missing key(s) {missing_keys} in {condition}.
        See https://docs.haystack.deepset.ai/docs/metadata-filtering for details on filters syntax."""
        raise AzureAISearchDocumentStoreFilterError(msg)

    operator = condition["operator"]
    if operator not in LOGICAL_OPERATORS:
        msg = f"Unknown operator {operator}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    conditions = []
    for c in condition["conditions"]:
        # Recursively parse if the condition itself is a logical condition
        if isinstance(c, dict) and "operator" in c and c["operator"] in LOGICAL_OPERATORS:
            conditions.append(_parse_logical_condition(c))
        else:
            # Otherwise, parse it as a comparison condition
            conditions.append(_parse_comparison_condition(c))

    # Format the result based on the operator
    if operator == "NOT":
        return f"not ({' and '.join([f'({c})' for c in conditions])})"
    else:
        return f" {LOGICAL_OPERATORS[operator]} ".join([f"({c})" for c in conditions])


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
    missing_keys = [key for key in ("field", "operator", "value") if key not in condition]
    if missing_keys:
        msg = f"""Missing key(s) {missing_keys} in {condition}.
        See https://docs.haystack.deepset.ai/docs/metadata-filtering for details on filters syntax."""
        raise AzureAISearchDocumentStoreFilterError(msg)

    # Remove the "meta." prefix from the field name if present
    field = condition["field"][5:] if condition["field"].startswith("meta.") else condition["field"]
    operator = condition["operator"]
    value = "null" if condition["value"] is None else condition["value"]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown operator {operator}. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise AzureAISearchDocumentStoreFilterError(msg)

    return COMPARISON_OPERATORS[operator](field, value)


def _eq(field: str, value: Any) -> str:
    return f"{field} eq '{value}'" if isinstance(value, str) and value != "null" else f"{field} eq {value}"


def _ne(field: str, value: Any) -> str:
    return f"not ({field} eq '{value}')" if isinstance(value, str) and value != "null" else f"not ({field} eq {value})"


def _in(field: str, value: Any) -> str:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        msg = "Azure AI Search only supports a list of strings for 'in' comparators"
        raise AzureAISearchDocumentStoreFilterError(msg)
    values = ",".join(map(str, value))
    return f"search.in({field},'{values}',',')"


def _comparison_operator(field: str, value: Any, operator: str) -> str:
    _validate_type(value, operator)
    return f"{field} {operator} {value}"


def _validate_type(value: Any, operator: str) -> None:
    """Validates that the value is either an integer, float, or ISO 8601 string."""
    msg = f"Invalid value type for '{operator}' comparator. Supported types are: int, float, or ISO 8601 string."

    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except ValueError as e:
            raise AzureAISearchDocumentStoreFilterError(msg) from e
    elif not isinstance(value, (int, float)):
        raise AzureAISearchDocumentStoreFilterError(msg)


COMPARISON_OPERATORS = {
    "==": _eq,
    "!=": _ne,
    "in": _in,
    ">": lambda f, v: _comparison_operator(f, v, "gt"),
    ">=": lambda f, v: _comparison_operator(f, v, "ge"),
    "<": lambda f, v: _comparison_operator(f, v, "lt"),
    "<=": lambda f, v: _comparison_operator(f, v, "le"),
}
