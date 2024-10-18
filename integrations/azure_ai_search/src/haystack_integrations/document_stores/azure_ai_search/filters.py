from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from haystack.utils import raise_on_invalid_filter_syntax

from .errors import AzureAISearchDocumentStoreFilterError

LOGICAL_OPERATORS = {"AND": "and", "OR": "or", "NOT": "not"}


def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters in Azure AI Search compatible filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise AzureAISearchDocumentStoreFilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)  # return a string
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    
    operator = condition["operator"]
    if operator not in LOGICAL_OPERATORS:
        msg = f"Unknown operator {operator}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    final_filter = f" {LOGICAL_OPERATORS[operator]} ".join([f"({c})" for c in conditions])
    return final_filter

    final_filter = ""
    for c in conditions[:-1]:
        final_filter += f"({c}) {LOGICAL_OPERATORS[operator]} "

    return final_filter + "(" + conditions[-1] + ")"


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "field" not in condition:
        msg = f"'field' key missing in {condition}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    field: str = ""
    # remove the "meta." prefix from the field name
    if condition["field"].startswith("meta."):
        field = condition["field"][5:]
    else:
        field = condition["field"]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]
    if value is None:
        value = "null"

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown operator {operator}. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise AzureAISearchDocumentStoreFilterError(msg)
    return COMPARISON_OPERATORS[operator](field, value)


def _eq(field: str, value: Any) -> str:
    if value == "null":
        return f"{field} eq {value}"
    if isinstance(value, str):
        return f"{field} eq '{value}'"
    return f"{field} eq {value}"


def _ne(field: str, value: Any) -> str:

    if isinstance(value, str):
        return f"not ({field} eq '{value}')"
    return f"not ({field} eq {value})"


def _gt(field: str, value: Any) -> str:
    return f"{field} gt {value}"


def _ge(field: str, value: Any) -> str:
    return f"{field} ge {value}"


def _lt(field: str, value: Any) -> str:
    return f"{field} lt {value}"


def _le(field: str, value: Any) -> str:
    return f"{field} le {value}"


def _in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = "Value must be a list when using 'in' comparators"
        raise AzureAISearchDocumentStoreFilterError(msg)
    elif any([not isinstance(v, str) for v in value]):
        msg = "Azure AI Search only supports string values for 'in' comparators"
        raise AzureAISearchDocumentStoreFilterError(msg)
    values = ", ".join([str(v) for v in value])
    return f"search.in({field},'{values}')"


COMPARISON_OPERATORS = {
    "==": _eq,
    "!=": _ne,
    ">": _gt,
    ">=": _ge,
    "<": _lt,
    "<=": _le,
    "in": _in,
    #  "not in": "$nin",
}
