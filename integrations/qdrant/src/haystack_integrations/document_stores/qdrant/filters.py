from datetime import datetime
from typing import List, Optional, Union

from haystack.utils.filters import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from qdrant_client.http import models

from .converters import convert_id

COMPARISON_OPERATORS = COMPARISON_OPERATORS.keys()
LOGICAL_OPERATORS = LOGICAL_OPERATORS.keys()


def convert_filters_to_qdrant(
    filter_term: Optional[Union[List[dict], dict, models.Filter]] = None, is_parent_call: bool = True
) -> Optional[models.Filter]:
    """Converts Haystack filters to the format used by Qdrant."""

    if isinstance(filter_term, models.Filter):
        return filter_term
    if not filter_term:
        return None

    must_clauses, should_clauses, must_not_clauses, conditions, qdrant_filters, same_level_operators = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    same_operators_flag = False  # Flag the
    if isinstance(filter_term, dict):
        filter_term = [filter_term]

    for item in filter_term:
        operator = item.get("operator")

        # Check for same operators on the same level
        if operator in same_level_operators and operator in LOGICAL_OPERATORS:
            same_operators_flag = True
        else:
            same_level_operators.append(operator)
        if operator is None:
            msg = "Operator not found in filters"
            raise FilterError(msg)

        if operator in LOGICAL_OPERATORS and "conditions" not in item:
            msg = f"'conditions' not found for '{operator}'"
            raise FilterError(msg)
        qdrant_filter = []

        if operator in LOGICAL_OPERATORS:
            qdrant_filter = convert_filters_to_qdrant(item.get("conditions", []), is_parent_call=False)
            if operator == "AND":
                if same_operators_flag:
                    must_clauses = [must_clauses, qdrant_filter]
                else:
                    must_clauses.extend(qdrant_filter)
            elif operator == "OR":
                if same_operators_flag:
                    should_clauses = [should_clauses, qdrant_filter]
                else:
                    should_clauses.extend(qdrant_filter)
            elif operator == "NOT":
                if same_operators_flag:
                    must_not_clauses = [must_not_clauses, qdrant_filter]
                else:
                    must_not_clauses.extend(qdrant_filter)
        elif operator in COMPARISON_OPERATORS:
            field = item.get("field")
            value = item.get("value")
            if field is None or value is None:
                msg = f"'field' or 'value' not found for '{operator}'"
                raise FilterError(msg)

            parsed_conditions = _parse_comparison_operation(comparison_operation=operator, key=field, value=value)
            if len(parsed_conditions) == 1 and isinstance(parsed_conditions[0], models.Filter):
                qdrant_filters.extend(parsed_conditions)
                continue
            else:
                conditions.extend(parsed_conditions)

        else:
            msg = f"Unknown operator {operator} used in filters"
            raise FilterError(msg)

    payload_filter = []
    if same_operators_flag:
        if any(isinstance(i, list) for i in must_clauses):
            for i in must_clauses:
                payload_filter = models.Filter(
                    must=i or None,
                    should=should_clauses or None,
                    must_not=must_not_clauses or None,
                )
                qdrant_filters.append(payload_filter)
        if any(isinstance(i, list) for i in should_clauses):
            for clause in should_clauses:
                payload_filter = models.Filter(
                    must=must_clauses or None,
                    should=clause or None,
                    must_not=must_not_clauses or None,
                )
                qdrant_filters.append(payload_filter)
        if any(isinstance(i, list) for i in must_not_clauses):
            for clause in must_not_clauses:
                payload_filter = models.Filter(
                    must=must_clauses or None,
                    should=should_clauses or None,
                    must_not=clause or None,
                )
                qdrant_filters.append(payload_filter)

        if not is_parent_call:
            return qdrant_filters

    payload_filter = models.Filter(
        must=must_clauses or None,
        should=should_clauses or None,
        must_not=must_not_clauses or None,
    )
    if is_parent_call:
        if conditions:
            must_clauses.extend(conditions)
        payload_filter = models.Filter(
            must=must_clauses or None,
            should=should_clauses or None,
            must_not=must_not_clauses or None,
        )
        if len(qdrant_filters) == 1:
            return qdrant_filters[0]
        else:
            return payload_filter
    elif conditions:
        qdrant_filters.extend(conditions)

    if must_clauses or should_clauses or must_not_clauses:
        qdrant_filters.append(payload_filter)

    return qdrant_filters


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
    for filter_part in filter_parts.items():
        if not filter_part:
            continue

        subfilter = filter_part[0]
        if not isinstance(subfilter, models.Filter):
            # The inner statement is a simple condition like models.FieldCondition
            # so it cannot be simplified.
            continue

        # if subfilter.must:
        # return models.Filter(**{part_name: subfilter.must})

    return payload_filter


def is_datetime_string(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
