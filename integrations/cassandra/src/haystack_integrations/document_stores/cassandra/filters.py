"""
Filter utilities for Cassandra Document Store.

Provides filter parsing and conversion for Cassandra CQL queries.
"""

from typing import Any

from haystack.dataclasses import Filter


def convert_filter_to_cql(
    filter_: Filter | None, param_prefix: str = "filter"
) -> tuple[str, dict[str, Any]]:
    """
    Convert Haystack Filter to CQL WHERE clause and parameters.

    :param filter_: Haystack Filter object.
    :param param_prefix: Prefix for parameter names.
    :returns: Tuple of (WHERE clause, parameters dict).
    """
    if not filter_:
        return "", {}

    conditions = []
    params = {}
    _convert_filter_condition(filter_, conditions, params, param_prefix)

    where_clause = " AND ".join(conditions)
    return where_clause, params


def _convert_filter_condition(
    filter_: Filter,
    conditions: list[str],
    params: dict[str, Any],
    param_prefix: str,
    param_counter: int = 0,
) -> int:
    """
    Recursively convert filter conditions.

    :param filter_: Filter to convert.
    :param conditions: List to append conditions to.
    :param params: Dict to add parameters to.
    :param param_prefix: Prefix for parameter names.
    :param param_counter: Current parameter counter.
    :returns: Updated parameter counter.
    """
    if filter_.operator in ("AND", "OR"):
        sub_conditions = []
        for sub_filter in filter_.conditions:
            param_counter = _convert_filter_condition(
                sub_filter, sub_conditions, params, param_prefix, param_counter
            )
        if sub_conditions:
            joined = f" {filter_.operator} ".join(sub_conditions)
            conditions.append(f"({joined})")
    else:
        field = filter_.field
        operator = _convert_operator(filter_.operator)
        param_name = f"{param_prefix}_{param_counter}"
        param_counter += 1

        # Handle special case for metadata fields
        if field.startswith("meta."):
            # For metadata, we need to extract from JSON
            # This is a simplified approach - in production use Cassandra's JSON functions
            conditions.append(f"metadata->>'{field[5:]}' {operator} %({param_name})s")
        else:
            conditions.append(f"{field} {operator} %({param_name})s")

        params[param_name] = filter_.value

    return param_counter


def _convert_operator(operator: str) -> str:
    """Convert Haystack operator to CQL operator."""
    mapping = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "in": "IN",
        "not in": "NOT IN",
    }
    return mapping.get(operator, "=")
