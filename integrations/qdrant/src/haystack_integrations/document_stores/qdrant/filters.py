from datetime import datetime
from typing import List, Optional, Union

from haystack.utils.filters import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from qdrant_client.http import models

from .converters import convert_id

COMPARISON_OPERATORS = COMPARISON_OPERATORS.keys()
LOGICAL_OPERATORS = LOGICAL_OPERATORS.keys()


def convert_filters_to_qdrant(
    filter_term: Optional[Union[List[dict], dict, models.Filter]] = None,
) -> Optional[models.Filter]:
    """Converts Haystack filters to the format used by Qdrant."""
    if isinstance(filter_term, models.Filter):
        return filter_term
    if not filter_term:
        return None

    must_clauses, should_clauses, must_not_clauses = [], [], []

    if isinstance(filter_term, dict):
        filter_term = [filter_term]

    for item in filter_term:
        operator = item.get("operator")
        if operator is None:
            msg = "Operator not found in filters"
            raise FilterError(msg)

        if operator in LOGICAL_OPERATORS and "conditions" not in item:
            msg = f"'conditions' not found for '{operator}'"
            raise FilterError(msg)

        if operator == "AND":
            must_clauses.append(convert_filters_to_qdrant(item.get("conditions", [])))
        elif operator == "OR":
            should_clauses.append(convert_filters_to_qdrant(item.get("conditions", [])))
        elif operator == "NOT":
            must_not_clauses.append(convert_filters_to_qdrant(item.get("conditions", [])))
        elif operator in COMPARISON_OPERATORS:
            field = item.get("field")
            value = item.get("value")
            if field is None or value is None:
                msg = f"'field' or 'value' not found for '{operator}'"
                raise FilterError(msg)

            must_clauses.extend(_parse_comparison_operation(comparison_operation=operator, key=field, value=value))
        else:
            msg = f"Unknown operator {operator} used in filters"
            raise FilterError(msg)

    payload_filter = models.Filter(
        must=must_clauses or None,
        should=should_clauses or None,
        must_not=must_not_clauses or None,
    )

    filter_result = _squeeze_filter(payload_filter)

    return filter_result


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
        models.FieldCondition(key=key, match=models.MatchText(text=value))
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


def _build_has_id_condition(id_values: List[models.ExtendedPointId]) -> models.HasIdCondition:
    return models.HasIdCondition(
        has_id=[
            # Ids are converted into their internal representation
            convert_id(item)
            for item in id_values
        ]
    )


def _squeeze_filter(payload_filter: models.Filter) -> models.Filter:
    """
    Simplify given payload filter, if the nested structure might be unnested.
    That happens if there is a single clause in that filter.
    :param payload_filter:
    :returns:
    """
    filter_parts = {
        "must": payload_filter.must,
        "should": payload_filter.should,
        "must_not": payload_filter.must_not,
    }

    total_clauses = sum(len(x) for x in filter_parts.values() if x is not None)
    if total_clauses == 0 or total_clauses > 1:
        return payload_filter

    # Payload filter has just a single clause provided (either must, should
    # or must_not). If that single clause is also of a models.Filter type,
    # then it might be returned instead.
    for part_name, filter_part in filter_parts.items():
        if not filter_part:
            continue

        subfilter = filter_part[0]
        if not isinstance(subfilter, models.Filter):
            # The inner statement is a simple condition like models.FieldCondition
            # so it cannot be simplified.
            continue

        if subfilter.must:
            return models.Filter(**{part_name: subfilter.must})

    return payload_filter


def is_datetime_string(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
