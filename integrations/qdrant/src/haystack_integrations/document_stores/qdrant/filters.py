from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from haystack.utils.filters import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from qdrant_client.http import models


def convert_filters_to_qdrant(
    filter_term: Optional[Union[List[Dict[str, Any]], Dict[str, Any], models.Filter]] = None,
) -> Optional[models.Filter]:
    """Converts Haystack filters to the format used by Qdrant.

    :param filter_term: the haystack filter to be converted to qdrant.
    :returns: a single Qdrant Filter or None.
    :raises FilterError: If invalid filter criteria is provided.
    """
    if isinstance(filter_term, models.Filter):
        return filter_term
    if not filter_term:
        return None

    if isinstance(filter_term, dict):
        filter_term = [filter_term]

    conditions = _process_filter_items(filter_term)

    return _build_final_filter(conditions)


def _process_filter_items(filter_items: List[Dict[str, Any]]) -> List[models.Condition]:
    """Process a list of filter items and return all conditions."""
    all_conditions: List[models.Condition] = []

    for item in filter_items:
        operator = item.get("operator")
        if operator is None:
            msg = "Operator not found in filters"
            raise FilterError(msg)

        if operator in LOGICAL_OPERATORS:
            condition = _process_logical_operator(item)
            if condition:
                all_conditions.append(condition)
        elif operator in COMPARISON_OPERATORS:
            condition = _process_comparison_operator(item)
            if condition:
                all_conditions.append(condition)
        else:
            msg = f"Unknown operator {operator} used in filters"
            raise FilterError(msg)

    return all_conditions


def _process_logical_operator(item: Dict[str, Any]) -> Optional[models.Condition]:
    """Process a logical operator (AND, OR, NOT) and return the corresponding condition."""
    operator = item["operator"]
    conditions = item.get("conditions")

    if not conditions:
        msg = f"'conditions' not found for '{operator}'"
        raise FilterError(msg)

    # Recursively process nested conditions
    nested_conditions = _process_filter_items(conditions)

    if not nested_conditions:
        return None

    # Build the appropriate filter based on operator
    if operator == "AND":
        return models.Filter(must=nested_conditions)
    elif operator == "OR":
        return models.Filter(should=nested_conditions)
    elif operator == "NOT":
        return models.Filter(must_not=nested_conditions)

    return None


def _process_comparison_operator(item: Dict[str, Any]) -> Optional[models.Condition]:
    """Process a comparison operator and return the corresponding condition."""
    operator = item["operator"]
    field = item.get("field")
    value = item.get("value")

    if field is None or value is None:
        msg = f"'field' or 'value' not found for '{operator}'"
        raise FilterError(msg)

    return _build_comparison_condition(operator, field, value)


def _build_final_filter(conditions: List[models.Condition]) -> Optional[models.Filter]:
    """Build the final filter from a list of conditions."""
    if not conditions:
        return None

    if len(conditions) == 1:
        # If single condition and it's already a Filter, return it
        if isinstance(conditions[0], models.Filter):
            return conditions[0]
        # Otherwise wrap it in a Filter
        return models.Filter(must=[conditions[0]])

    # Multiple conditions - combine with AND logic
    return models.Filter(must=conditions)


def _build_comparison_condition(operator: str, key: str, value: Any) -> models.Condition:
    """Build a comparison condition based on operator, key, and value."""
    condition_builders: Dict[str, Callable[[str, Any], models.Condition]] = {
        "==": _build_eq_condition,
        "in": _build_in_condition,
        "!=": _build_ne_condition,
        "not in": _build_nin_condition,
        ">": _build_gt_condition,
        ">=": _build_gte_condition,
        "<": _build_lt_condition,
        "<=": _build_lte_condition,
    }

    builder = condition_builders.get(operator)
    if builder is None:
        msg = f"Unknown operator {operator} used in filters"
        raise FilterError(msg)

    return builder(key, value)


def _build_eq_condition(key: str, value: models.ValueVariants) -> models.Condition:
    if isinstance(value, str) and " " in value:
        return models.FieldCondition(key=key, match=models.MatchText(text=value))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _build_in_condition(key: str, value: List[models.ValueVariants]) -> models.Condition:
    if not isinstance(value, list):
        msg = f"Value {value} is not a list"
        raise FilterError(msg)
    return models.Filter(
        should=[
            (
                models.FieldCondition(key=key, match=models.MatchText(text=item))
                if isinstance(item, str) and " " not in item
                else models.FieldCondition(key=key, match=models.MatchValue(value=item))
            )
            for item in value
        ]
    )


def _build_ne_condition(key: str, value: models.ValueVariants) -> models.Condition:
    return models.Filter(
        must_not=[
            (
                models.FieldCondition(key=key, match=models.MatchText(text=value))
                if isinstance(value, str) and " " not in value
                else models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        ]
    )


def _build_nin_condition(key: str, value: List[models.ValueVariants]) -> models.Condition:
    if not isinstance(value, list):
        msg = f"Value {value} is not a list"
        raise FilterError(msg)
    return models.Filter(
        must_not=[
            (
                models.FieldCondition(key=key, match=models.MatchText(text=item))
                if isinstance(item, str) and " " in item
                else models.FieldCondition(key=key, match=models.MatchValue(value=item))
            )
            for item in value
        ]
    )


def _build_lt_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        dt_value = datetime.fromisoformat(value)
        return models.FieldCondition(key=key, range=models.DatetimeRange(lt=dt_value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(lt=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_lte_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        dt_value = datetime.fromisoformat(value)
        return models.FieldCondition(key=key, range=models.DatetimeRange(lte=dt_value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(lte=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_gt_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        dt_value = datetime.fromisoformat(value)
        return models.FieldCondition(key=key, range=models.DatetimeRange(gt=dt_value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(gt=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_gte_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        dt_value = datetime.fromisoformat(value)
        return models.FieldCondition(key=key, range=models.DatetimeRange(gte=dt_value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(gte=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def is_datetime_string(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
