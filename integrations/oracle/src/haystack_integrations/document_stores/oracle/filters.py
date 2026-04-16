# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, ClassVar

from haystack.errors import FilterError

_RANGE_OPS = {">", ">=", "<", "<="}


class FilterTranslator:
    """
    Translates Haystack 2.x filter dicts into Oracle SQL WHERE fragments.

    Example input:
        {"operator": "AND", "conditions": [
            {"field": "meta.author", "operator": "==", "value": "Alice"},
            {"field": "meta.year",   "operator": ">",  "value": 2020},
        ]}

    Example output SQL fragment:
        (JSON_VALUE(metadata, '$.author') = :p0
         AND TO_NUMBER(JSON_VALUE(metadata, '$.year')) > :p1)

    Params dict is mutated in-place; caller passes an empty dict and uses it
    for cursor.execute / cursor.executemany bindings.
    """

    _OP_MAP: ClassVar[dict[str, str]] = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }

    def translate(self, filters: dict[str, Any], params: dict[str, Any], counter: list[int]) -> str:
        """
        Translate the given filter dict into an SQL fragment, adding any necessary parameters to the params dict.
        """
        op = filters.get("operator")

        # Logical nodes
        if op in ("AND", "OR", "NOT"):
            if "conditions" not in filters:
                msg = f"'conditions' key missing in logical filter: {filters}"
                raise FilterError(msg)
            if op == "AND":
                parts = [self.translate(c, params, counter) for c in filters["conditions"]]
                return "(" + " AND ".join(parts) + ")"
            if op == "OR":
                parts = [self.translate(c, params, counter) for c in filters["conditions"]]
                return "(" + " OR ".join(parts) + ")"
            # NOT
            inner = self.translate(filters["conditions"][0], params, counter)
            return f"(NOT {inner})"

        # Comparison leaf — validate required keys first
        if "field" not in filters:
            msg = f"'field' key missing in comparison filter: {filters}"
            raise FilterError(msg)
        if "operator" not in filters:
            msg = f"'operator' key missing in comparison filter: {filters}"
            raise FilterError(msg)
        if "value" not in filters:
            msg = f"'value' key missing in comparison filter: {filters}"
            raise FilterError(msg)

        if not isinstance(op, str) or op not in {*self._OP_MAP, "in", "not in"}:
            msg = f"Unsupported filter operator: {op!r}"
            raise FilterError(msg)

        field: str = filters["field"]
        value: Any = filters["value"]

        if op in ("in", "not in"):
            if not isinstance(value, list):
                msg = f"'in' / 'not in' filter values must be a list, got {type(value).__name__!r}"
                raise FilterError(msg)
            col = FilterTranslator._field_to_sql(field, value[0] if value else value)
            placeholders = []
            for v in value:
                pname = f"p{counter[0]}"
                counter[0] += 1
                params[pname] = v
                placeholders.append(f":{pname}")
            if op == "in":
                return f"{col} IN ({', '.join(placeholders)})"
            # NOT IN: include rows where the column is NULL, matching Python's
            # `None not in [...]` == True semantics.
            return f"({col} IS NULL OR {col} NOT IN ({', '.join(placeholders)}))"

        if op in _RANGE_OPS and (isinstance(value, (str, list)) and not _is_iso_date(value)):
            msg = f"Operator {op!r} requires a numeric or ISO-date value, got {type(value).__name__!r}: {value!r}"
            raise FilterError(msg)

        col = FilterTranslator._field_to_sql(field, value)

        # NULL-safe equality / inequality
        if op == "==" and value is None:
            return f"{col} IS NULL"
        if op == "!=" and value is None:
            return f"{col} IS NOT NULL"

        pname = f"p{counter[0]}"
        counter[0] += 1
        params[pname] = value

        if op == "!=":
            # Include rows where the column is NULL, matching Python's
            # `None != x` == True semantics.
            return f"({col} != :{pname} OR {col} IS NULL)"

        sql_op = self._OP_MAP[op]
        return f"{col} {sql_op} :{pname}"

    @staticmethod
    def _field_to_sql(field: str, value: Any) -> str:
        if field == "id":
            return "id"
        if field == "content":
            return "text"
        if field.startswith("meta."):
            key = field[len("meta.") :]
            json_path = f"JSON_VALUE(metadata, '$.{key}')"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"TO_NUMBER({json_path})"
            return json_path
        # Fallback: treat as top-level JSON key
        json_path = f"JSON_VALUE(metadata, '$.{field}')"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"TO_NUMBER({json_path})"
        return json_path


def _is_iso_date(value: Any) -> bool:
    """Return True if *value* is a string that Python recognises as a valid ISO-8601 datetime."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
