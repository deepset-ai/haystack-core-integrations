# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for translating Haystack filters into Mem0 Platform filters.

Mem0 Platform `search` and `get_all` expect entity IDs such as `user_id`, `agent_id`, `app_id`, and `run_id`
inside the `filters` object. They also reserve a fixed set of top-level filter fields such as `created_at`,
`categories`, `keywords`, and `memory_ids`.

See:
- https://docs.mem0.ai/api-reference/memory/search-memories
- https://docs.mem0.ai/api-reference/memory/get-memories
- https://docs.mem0.ai/platform/features/v2-memory-filters

Haystack users usually think of arbitrary fields as metadata filters. To keep that API natural while sending valid
Mem0 filters, fields that are not native Mem0 filter fields are nested under `metadata`. For example,
`{"field": "category", "operator": "==", "value": "work"}` becomes `{"metadata": {"category": "work"}}`.
"""

from typing import Any

from haystack.errors import FilterError

# Keep this list aligned with the top-level filter keys documented in:
# https://docs.mem0.ai/api-reference/memory/search-memories
# https://docs.mem0.ai/platform/features/v2-memory-filters
_TOP_LEVEL_FILTER_FIELDS = {
    "user_id",
    "agent_id",
    "app_id",
    "run_id",
    "created_at",
    "updated_at",
    "timestamp",
    "categories",
    "metadata",
    "keywords",
    "feedback",
    "feedback_reason",
    "memory_ids",
}


def _build_search_filters(
    *,
    filters: dict[str, Any] | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
) -> dict[str, Any]:
    """
    Build Mem0 search filters from explicit IDs and optional Haystack-style filters.

    Mem0 `search` and `get_all` expect entity IDs inside the `filters` object.
    This helper keeps the public store API convenient while ensuring IDs and custom filters are both applied.
    It combines everything as Haystack-style filters first, then normalizes the combined filter to Mem0 format.
    Fields that are not native Mem0 filter fields are treated as Mem0 metadata fields.

    Mem0 filter reference:
    - https://docs.mem0.ai/api-reference/memory/search-memories
    - https://docs.mem0.ai/api-reference/memory/get-memories

    :param filters: Haystack-style filters to combine with IDs.
    :param user_id: User ID to scope the search.
    :param run_id: Run ID to scope the search.
    :param agent_id: Agent ID to scope the search.
    :param app_id: App ID to scope the search.
    :returns: Mem0-compatible filters.
    :raises ValueError: If neither filters nor an entity ID is provided.
    """
    conditions = [
        {"field": key, "operator": "==", "value": value}
        for key, value in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id, "app_id": app_id}.items()
        if value
    ]

    if filters:
        if filters.get("operator", "").upper() == "AND" and "conditions" in filters:
            conditions.extend(filters["conditions"])
        else:
            conditions.append(filters)

    if not conditions:
        msg = "Either filters or at least one of user_id, run_id, agent_id, or app_id must be provided."
        raise ValueError(msg)

    combined_filter = conditions[0] if len(conditions) == 1 else {"operator": "AND", "conditions": conditions}
    return normalize_filters(combined_filter)


def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Haystack-style filters to the Mem0 filter format.

    Mem0 Platform supports logical filters (`AND`, `OR`, `NOT`) and comparison operators on native fields.
    It also supports metadata filters under the top-level `metadata` key.
    See https://docs.mem0.ai/platform/features/v2-memory-filters for the list of fields and operators.

    :param filters: Haystack filter dictionary.
    :returns: Equivalent Mem0 filter dictionary. Fields that are not native Mem0 filter fields are nested under
        `metadata`.
    :raises FilterError: If the filter structure or operators are invalid.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary."
        raise FilterError(msg)
    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a Haystack logical condition into Mem0's logical filter shape.

    Mem0 documents `AND`, `OR`, and `NOT` as supported logical operators:
    https://docs.mem0.ai/platform/features/v2-memory-filters
    """
    if "operator" not in condition:
        msg = f"'operator' key missing in: {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in: {condition}"
        raise FilterError(msg)

    operator = condition["operator"].upper()
    if operator not in ("AND", "OR", "NOT"):
        msg = f"Unsupported logical operator: {operator!r}. Use AND, OR, or NOT."
        raise FilterError(msg)

    converted = [_convert(c) for c in condition["conditions"]]
    return {operator: converted}


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a Haystack comparison condition into Mem0's native or metadata filter shape.

    Native Mem0 fields are kept at the top level. All other fields are treated as metadata fields because Mem0's
    Platform API expects metadata filters under the `metadata` key:
    https://docs.mem0.ai/platform/features/v2-memory-filters
    """
    if "field" not in condition:
        msg = f"'field' key missing in: {condition}"
        raise FilterError(msg)
    if "operator" not in condition:
        msg = f"'operator' key missing in: {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in: {condition}"
        raise FilterError(msg)

    field: str = condition["field"]
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if field not in _TOP_LEVEL_FILTER_FIELDS:
        return _parse_metadata_comparison_condition(field, operator, value)

    op_map = {
        "==": lambda f, v: {f: v},
        "!=": lambda f, v: {f: {"ne": v}},
        ">": lambda f, v: {f: {"gt": v}},
        ">=": lambda f, v: {f: {"gte": v}},
        "<": lambda f, v: {f: {"lt": v}},
        "<=": lambda f, v: {f: {"lte": v}},
        "in": lambda f, v: {f: {"in": v if isinstance(v, list) else [v]}},
        "not in": lambda f, v: {f: {"ne": v}},
        "contains": lambda f, v: {f: {"contains": v}},
        "icontains": lambda f, v: {f: {"icontains": v}},
    }

    if operator not in op_map:
        msg = f"Unsupported filter operator: {operator!r}."
        raise FilterError(msg)

    return op_map[operator](field, value)


def _parse_metadata_comparison_condition(field: str, operator: str, value: Any) -> dict[str, Any]:
    """
    Convert a Haystack metadata comparison into Mem0's `metadata` filter shape.

    Mem0 metadata filters support bare equality, `ne`, and `contains`; operators such as `in`, `gt`, and `lt` are not
    supported for metadata filters.
    For multi-value metadata checks, Mem0 recommends composing equality clauses with `OR`.

    See https://docs.mem0.ai/platform/features/v2-memory-filters
    """
    metadata_field = field.removeprefix("metadata.")
    op_map = {
        "==": lambda f, v: {"metadata": {f: v}},
        "!=": lambda f, v: {"metadata": {f: {"ne": v}}},
        "contains": lambda f, v: {"metadata": {f: {"contains": v}}},
    }

    if operator not in op_map:
        msg = (
            f"Unsupported metadata filter operator: {operator!r}. "
            "Mem0 metadata filters support ==, !=, and contains. "
            "For multi-value metadata filters, combine equality conditions with OR."
        )
        raise FilterError(msg)

    return op_map[operator](metadata_field, value)


def _convert(node: dict[str, Any]) -> dict[str, Any]:
    if "field" in node:
        return _parse_comparison_condition(node)
    return _parse_logical_condition(node)
