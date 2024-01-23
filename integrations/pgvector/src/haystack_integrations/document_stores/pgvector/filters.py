# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict

from haystack.errors import FilterError
from pandas import DataFrame
from datetime import datetime

from psycopg.sql import SQL
from psycopg.types.json import Jsonb


def _build_where_clause(filters: Dict[str, Any], cursor) -> str:

    normalized_filters = _normalize_filters(filters)
    print("normalized_filters", normalized_filters)
    where_clause = SQL(" WHERE ")

    if isinstance(normalized_filters, dict):
        top_level_operator = list(normalized_filters.keys())[0]
        conditions = normalized_filters[top_level_operator]
    else:
        top_level_operator = None
        conditions = normalized_filters
    # condtions = condition[top_level_operator]
        
    if top_level_operator == "NOT":
        where_clause += SQL("NOT (")

    params = ()
    skip = 0
    for i, condition in enumerate(conditions):
        print("condition", condition)
        field,op,value = condition[0] if isinstance(condition, list) else condition

        where_clause+= SQL("{field}{operator}").format(field=SQL(field),
                                                                     operator=SQL(op))
               
        where_clause_str = where_clause.as_string(cursor)                                                     
        if where_clause_str.count("%s") + skip < i + 1:
            if where_clause_str.endswith("IS NULL"):
                skip += 1
            else:
                where_clause += SQL(" %s")

      
        if value!= "nonetype":
            params = params + (value,)

        # if "dataframe" in field:
        #     where_clause += SQL("::jsonb")            

        if i< len(conditions) - 1:
            if top_level_operator == "OR":
                where_clause += SQL(" OR ")
            else:
                where_clause += SQL(" AND ")

    if top_level_operator == "NOT":
        where_clause += SQL(")")

    return where_clause, params

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

    if operator in LOGICAL_OPERATORS:
        return {LOGICAL_OPERATORS[operator]: conditions}

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
        k,_,v = field.partition(".")
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
            if not(isinstance(value[0], str)):
                type_value = python_types_to_pg_types[type(value[0])]           
        
        if type_value:
            field = f"({field})::{type_value}"

    return [COMPARISON_OPERATORS[operator](field, value)]


def _equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return field, " IS NULL", "nonetype"

    return field, "=", value


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    return field, " IS DISTINCT FROM ", value


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

    return field, ">", value


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

    return field, ">=", value


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

    return field, "<", value


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

    return field, "<=", value


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    return field+ " IS NULL OR "+field, "!= ALL(%s)", [value]


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    return field, "= ANY(%s)", [value]


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