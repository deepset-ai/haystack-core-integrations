# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from itertools import chain
from typing import Any, Literal

from haystack.errors import FilterError
from psycopg.sql import SQL, Composable, Composed, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

# we need this mapping to cast meta values to the correct type,
# since they are stored in the JSONB field as strings.
# this dict can be extended if needed
PYTHON_TYPES_TO_PG_TYPES = {
    int: "integer",
    float: "real",
    bool: "boolean",
}

NO_VALUE = object()  # unique sentinel; can never equal a user-supplied filter value

# Operators that match against a JSONB array stored in the meta field.
# They need the JSONB value (via the `->` accessor) as their left operand,
# unlike the scalar operators which access meta as text (via `->>`).
ARRAY_OPERATORS = ("array_contains", "array_overlaps")


def _validate_filters(filters: dict[str, Any] | None = None) -> None:
    """
    Validates the filters provided.
    """
    if filters:
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise TypeError(msg)
        if "operator" not in filters and "conditions" not in filters:
            msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
            raise ValueError(msg)


def _convert_filters_to_where_clause_and_params(
    filters: dict[str, Any], operator: Literal["WHERE", "AND"] = "WHERE"
) -> tuple[Composed, tuple]:
    """
    Convert Haystack filters to a WHERE clause and a tuple of params to query PostgreSQL.
    """
    if "field" in filters:
        query, values = _parse_comparison_condition(filters)
    else:
        query, values = _parse_logical_condition(filters)

    where_clause = SQL(f" {operator} ") + query
    params = tuple(value for value in values if value is not NO_VALUE)

    return where_clause, params


def _parse_logical_condition(condition: dict[str, Any]) -> tuple[Composed, list[Any]]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR"]:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR'"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    conditions = []
    for c in condition["conditions"]:
        if "field" in c:
            query, vals = _parse_comparison_condition(c)
        else:
            query, vals = _parse_logical_condition(c)
        conditions.append((query, vals))

    query_parts, values = [], []
    for c in conditions:
        query_parts.append(c[0])
        values.append(c[1])
    if isinstance(values[0], list):
        values = list(chain.from_iterable(values))

    if operator == "AND":
        sql_query = SQL("(") + SQL(" AND ").join(query_parts) + SQL(")")
    else:  # operator == "OR"
        sql_query = SQL("(") + SQL(" OR ").join(query_parts) + SQL(")")

    return sql_query, values


def _parse_comparison_condition(condition: dict[str, Any]) -> tuple[Composed, list[Any]]:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]

    if field.startswith("meta."):
        # array operators need the JSONB value (->), the other operators access meta as text (->>)
        sql_field: Composable = (
            _treat_meta_field_as_jsonb(field) if operator in ARRAY_OPERATORS else _treat_meta_field(field, value)
        )
    elif operator in ARRAY_OPERATORS:
        # top-level columns are never JSONB arrays, so these operators would render SQL that the
        # database rejects with an opaque UndefinedFunction error. Fail with a FilterError instead.
        msg = (
            f"Operator '{operator}' is only supported on meta fields, but got field '{field}'. "
            f"Prefix the field name with 'meta.' to filter on an array stored in a document's metadata."
        )
        raise FilterError(msg)
    else:
        sql_field = Identifier(field)

    sql_expr, value = COMPARISON_OPERATORS[operator](sql_field, value)
    return sql_expr, [value]


def _treat_meta_field(field: str, value: Any) -> Composed:
    """
    Internal method that returns a psycopg Composed object to make the meta JSONB field queryable safely.

    Uses psycopg.sql.Literal to embed the field name, preventing SQL injection
    via metadata field names without requiring regex validation.

    Use the ->> operator to access keys in the meta JSONB field.

    Examples:
    >>> _treat_meta_field(field="meta.number", value=9)
    Composed([SQL('(meta->>'), Literal('number'), SQL(')::integer')])

    >>> _treat_meta_field(field="meta.name", value="my_name")
    Composed([SQL('meta->>'), Literal('name')])

    """
    field_name = field.split(".", 1)[-1]

    # Use SQLLiteral to safely embed the field name as a SQL string literal,
    # preventing SQL injection via metadata field names.
    composed: Composed = SQL("meta->>") + SQLLiteral(field_name)

    # meta fields are stored as strings in the JSONB field,
    # so we need to cast them to the correct type
    type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value))
    if isinstance(value, list) and len(value) > 0:
        type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value[0]))

    if type_value:
        composed = SQL("(") + composed + SQL(f")::{type_value}")

    return composed


def _treat_meta_field_as_jsonb(field: str) -> Composed:
    """
    Internal method that returns a psycopg Composed object accessing a meta key as a JSONB value.

    Unlike `_treat_meta_field`, this uses the `->` operator (which returns a JSONB value) instead of
    `->>` (which returns text) and applies no type cast. The array operators need a JSONB left operand
    for their JSONB operators (`@>`, `?|`) to work.

    Uses psycopg.sql.Literal to embed the field name, preventing SQL injection via metadata field names.

    Example:
    >>> _treat_meta_field_as_jsonb(field="meta.tags")
    Composed([SQL('meta->'), Literal('tags')])
    """
    field_name = field.split(".", 1)[-1]
    return SQL("meta->") + SQLLiteral(field_name)


def _equal(field: Composable, value: Any) -> tuple[Composed, Any]:
    if value is None:
        return SQL("{} IS NULL").format(field), NO_VALUE
    return SQL("{} = %s").format(field), value


def _not_equal(field: Composable, value: Any) -> tuple[Composed, Any]:
    # we use IS DISTINCT FROM to correctly handle NULL values
    # (not handled by !=)
    return SQL("{} IS DISTINCT FROM %s").format(field), value


def _greater_than(field: Composable, value: Any) -> tuple[Composed, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return SQL("{} > %s").format(field), value


def _greater_than_equal(field: Composable, value: Any) -> tuple[Composed, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return SQL("{} >= %s").format(field), value


def _less_than(field: Composable, value: Any) -> tuple[Composed, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return SQL("{} < %s").format(field), value


def _less_than_equal(field: Composable, value: Any) -> tuple[Composed, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    return SQL("{} <= %s").format(field), value


def _not_in(field: Composable, value: Any) -> tuple[Composed, list]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in PgVector"
        raise FilterError(msg)
    return SQL("({} IS NULL OR {} != ALL(%s))").format(field, field), [value]


def _in(field: Composable, value: Any) -> tuple[Composed, list]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in PgVector"
        raise FilterError(msg)

    # see https://www.psycopg.org/psycopg3/docs/basic/adapt.html#lists-adaptation
    return SQL("{} = ANY(%s)").format(field), [value]


def _like(field: Composable, value: Any) -> tuple[Composed, Any]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'LIKE' "
        raise FilterError(msg)
    return SQL("{} LIKE %s").format(field), value


def _not_like(field: Composable, value: Any) -> tuple[Composed, Any]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'LIKE' "
        raise FilterError(msg)
    return SQL("{} NOT LIKE %s").format(field), value


def _validate_array_operator_value(field: Composable, value: Any, operator: str) -> None:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using '{operator}' comparator in PgVector"
        raise FilterError(msg)
    # an empty list has no useful match semantics: `@> '[]'` matches every array and `?| '{}'` matches
    # nothing, so the same empty filter would mean "everything" for one operator and "nothing" for the
    # other. Callers building the list dynamically should omit the condition instead.
    if not value:
        msg = f"{field}'s value must be a non-empty list when using '{operator}' comparator in PgVector"
        raise FilterError(msg)


def _array_contains(field: Composable, value: Any) -> tuple[Composed, Jsonb]:
    _validate_array_operator_value(field, value, "array_contains")
    # `jsonb @> jsonb` is true when the left array contains every element of the right array,
    # so the meta array must contain all of the given values. Containment compares elements by value,
    # so a non-array meta value (a bare string, an object) never matches a list of values.
    return SQL("{} @> %s").format(field), Jsonb(value)


def _array_overlaps(field: Composable, value: Any) -> tuple[Composed, list]:
    _validate_array_operator_value(field, value, "array_overlaps")
    # `jsonb ?| text[]` only matches string elements, so we reject non-string values to fail fast
    # instead of silently returning no results.
    if not all(isinstance(item, str) for item in value):
        msg = f"{field}'s value must be a list of strings when using 'array_overlaps' comparator in PgVector"
        raise FilterError(msg)
    # `?|` is true when a given string exists as a top-level key OR array element, so on its own it
    # also matches objects (by key) and bare strings. Restricting it to arrays with jsonb_typeof keeps
    # it in agreement with array_contains. The parentheses stop this AND from being split by an
    # enclosing OR (see _not_in for the same reason).
    return SQL("(jsonb_typeof({}) = 'array' AND {} ?| %s)").format(field, field), [value]


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
    "like": _like,
    "not like": _not_like,
    "array_contains": _array_contains,
    "array_overlaps": _array_overlaps,
}
