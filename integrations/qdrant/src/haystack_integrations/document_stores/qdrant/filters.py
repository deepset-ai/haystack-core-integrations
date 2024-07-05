from datetime import datetime
from typing import Any, List, Optional, Union

from haystack.utils.filters import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from qdrant_client.http import models

COMPARISON_OPERATORS = COMPARISON_OPERATORS.keys()
LOGICAL_OPERATORS = LOGICAL_OPERATORS.keys()


def convert_filters_to_qdrant(
    filter_term: Optional[Union[List[dict], dict, models.Filter]] = None, is_parent_call: bool = True
) -> Optional[Union[models.Filter, List[models.Filter]]]:
    """Converts Haystack filters to the format used by Qdrant."""

    if isinstance(filter_term, models.Filter):
        return filter_term
    if not filter_term:
        return None

    must_clauses: List[models.Filter] = []
    should_clauses: List[models.Filter] = []
    must_not_clauses: List[models.Filter] = []
    same_operator_flag = False  # For same operators on each level, we need nested clauses
    conditions, qdrant_filters, current_level_operators = (
        [],
        [],
        [],
    )

    if isinstance(filter_term, dict):
        filter_term = [filter_term]

    for item in filter_term:
        operator = item.get("operator")

        # Check for same operators on each level
        same_operator_flag = operator in current_level_operators and operator in LOGICAL_OPERATORS
        if not same_operator_flag:
            current_level_operators.append(operator)

        if operator is None:
            msg = "Operator not found in filters"
            raise FilterError(msg)

        if operator in LOGICAL_OPERATORS and "conditions" not in item:
            msg = f"'conditions' not found for '{operator}'"
            raise FilterError(msg)

        if operator in LOGICAL_OPERATORS:
            current_filter = convert_filters_to_qdrant(item.get("conditions", []), is_parent_call=False) or []

            if operator == "AND":
                must_clauses = [must_clauses, current_filter] if same_operator_flag else must_clauses + current_filter
            elif operator == "OR":
                should_clauses = (
                    [should_clauses, current_filter] if same_operator_flag else should_clauses + current_filter
                )
            elif operator == "NOT":
                must_not_clauses = (
                    [must_not_clauses, current_filter] if same_operator_flag else must_not_clauses + current_filter
                )

        elif operator in COMPARISON_OPERATORS:
            field = item.get("field")
            value = item.get("value")
            if field is None or value is None:
                msg = f"'field' or 'value' not found for '{operator}'"
                raise FilterError(msg)

            parsed_conditions = _parse_comparison_operation(comparison_operation=operator, key=field, value=value)

            # check if the parsed_conditions are models.Filter or models.Condition
            for condition in parsed_conditions:
                if isinstance(condition, models.Filter):
                    qdrant_filters.append(condition)
                else:
                    conditions.append(condition)

        else:
            msg = f"Unknown operator {operator} used in filters"
            raise FilterError(msg)

    if same_operator_flag:
        qdrant_filters = build_payload_for_same_operators(
            must_clauses, should_clauses, must_not_clauses, qdrant_filters
        )
        if not is_parent_call:
            return qdrant_filters
    elif must_clauses or should_clauses or must_not_clauses:
        qdrant_filters.append(build_payload(must_clauses, should_clauses, must_not_clauses))

    if is_parent_call:
        if conditions:
            must_clauses.extend(conditions)
        return (
            qdrant_filters[0]
            if len(qdrant_filters) == 1
            else build_payload(must_clauses, should_clauses, must_not_clauses)
        )

    if conditions:
        qdrant_filters.extend(conditions)

    return qdrant_filters


def build_payload(
    must_clauses: List[models.Condition],
    should_clauses: List[models.Condition],
    must_not_clauses: List[models.Condition],
) -> models.Filter:

    return models.Filter(
        must=must_clauses or None,
        should=should_clauses or None,
        must_not=must_not_clauses or None,
    )


def build_payload_for_same_operators(
    must_clauses: List[models.Condition],
    should_clauses: List[models.Condition],
    must_not_clauses: List[models.Condition],
    output_filter: List[Any],
) -> List[models.Filter]:

    clause_types = [
        (must_clauses, should_clauses, must_not_clauses),
        (should_clauses, must_clauses, must_not_clauses),
        (must_not_clauses, must_clauses, should_clauses),
    ]

    for clauses, arg1, arg2 in clause_types:
        if any(isinstance(i, list) for i in clauses):
            for clause in clauses:
                output_filter.append(build_payload(clause, arg1, arg2))

    return output_filter


def _parse_comparison_operation(
    comparison_operation: str, key: str, value: Union[dict, List, str, float]
) -> List[models.Condition]:
    conditions: List[models.Condition] = []

    condition_builder_mapping = {
        "==": _build_eq_condition,
        "in": _build_in_condition,
        "!=": _build_ne_condition,
        "not in": _build_nin_condition,
        ">": _build_gt_condition,
        ">=": _build_gte_condition,
        "<": _build_lt_condition,
        "<=": _build_lte_condition,
    }

    condition_builder = condition_builder_mapping.get(comparison_operation)

    if condition_builder is None:
        msg = f"Unknown operator {comparison_operation} used in filters"
        raise ValueError(msg)

    conditions.append(condition_builder(key, value))

    return conditions


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
                if isinstance(value, str) and " " in value
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
        return models.FieldCondition(key=key, range=models.DatetimeRange(lt=value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(lt=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_lte_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        return models.FieldCondition(key=key, range=models.DatetimeRange(lte=value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(lte=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_gt_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        return models.FieldCondition(key=key, range=models.DatetimeRange(gt=value))

    if isinstance(value, (int, float)):
        return models.FieldCondition(key=key, range=models.Range(gt=value))

    msg = f"Value {value} is not an int or float or datetime string"
    raise FilterError(msg)


def _build_gte_condition(key: str, value: Union[str, float, int]) -> models.Condition:
    if isinstance(value, str) and is_datetime_string(value):
        return models.FieldCondition(key=key, range=models.DatetimeRange(gte=value))

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
