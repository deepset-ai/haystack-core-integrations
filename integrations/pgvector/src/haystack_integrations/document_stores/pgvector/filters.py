# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Any, Dict

from haystack.errors import FilterError
from pandas import DataFrame
from psycopg.sql import SQL
from psycopg.types.json import Jsonb


def _build_where_clause(filters: Dict[str, Any], cursor) -> str:
    normalized_filters = _normalize_filters(filters)
    print("normalized_filters", normalized_filters)

    sql_query, params = normalized_filters
    if isinstance(params, list):
        params = tuple(params)
    else:
        params = (params,)

    if isinstance(sql_query, str):
        sql_query = SQL(sql_query)
    where_clause = SQL(" WHERE ") + sql_query
    # condtions = condition[top_level_operator]

    # if top_level_operator == "NOT":
    #     where_clause += SQL("NOT (")

    # params = ()
    # for i, condition in enumerate(conditions):
    #     print("condition", condition)
    #     query_part,value = condition[0] if isinstance(condition, list) else condition

    #     where_clause+= SQL(query_part)

    #     if value != "do_not_append_value":
    #         params = params + (value,)

    #     if i< len(conditions) - 1:
    #         if top_level_operator == "OR":
    #             where_clause += SQL(" OR ")
    #         else:
    #             where_clause += SQL(" AND ")

    # if top_level_operator == "NOT":
    #     where_clause += SQL(")")
    # if not isinstance(params, tuple):
    #     params = (params,)

    actual_params = ()
    for i, param in enumerate(params):
        if param != "no_value":
            actual_params = actual_params + (param,)

    return where_clause, actual_params


def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters in pgvector compatible filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    query_parts = []
    values = []
    for c in conditions:
        query_parts.append(SQL(c[0]))
        values.append(c[1])

    print("query_parts", query_parts)

    if operator == "AND":
        return SQL(" AND ").join(query_parts), values
    elif operator == "OR":
        return SQL(" OR ").join(query_parts), values
    elif operator == "NOT":
        query_parts = [SQL("NOT (") + query_part + SQL(")") for query_part in query_parts]
        return SQL(" AND ").join(query_parts), values
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _parse_logical_condition(condition)

    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    meta = False
    if field.startswith("meta."):
        meta = True
        # Remove the "meta." prefix if present.
        k, _, v = field.partition(".")
        field = f"{k}->>'{v}'"

    value: Any = condition["value"]
    if isinstance(value, DataFrame):
        value = value.to_json()
        field = f"({field})::jsonb"
        value = Jsonb(value)

    type_value = None
    if meta and not isinstance(value, (str, type(None))):
        python_types_to_pg_types = {
            int: "integer",
            float: "real",
            bool: "boolean",
        }

        if type(value) in python_types_to_pg_types:
            type_value = python_types_to_pg_types[type(value)]
        elif isinstance(value, list):
            if not (isinstance(value[0], str)):
                type_value = python_types_to_pg_types[type(value[0])]

        if type_value:
            field = f"({field})::{type_value}"

    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return f"{field} IS NULL", "no_value"

    return f"{field} = %s", value


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    return f"{field} IS DISTINCT FROM %s", value


def _greater_than(field: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} > %s", value


def _greater_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} >= %s", value


def _less_than(field: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} < %s", value


def _less_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, DataFrame, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} <= %s", value


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    return f"{field} IS NULL OR {field} != ALL(%s)", [value]


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    return f"{field} = ANY(%s)", [value]


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

LOGICAL_OPERATORS = {"AND": "AND", "OR": "OR", "NOT": "NOT"}
