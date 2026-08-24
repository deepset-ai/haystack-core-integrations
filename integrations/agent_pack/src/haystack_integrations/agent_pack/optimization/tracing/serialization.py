# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded serialization of Haystack span values, on top of Haystack's schema-aware value serialization."""

from typing import Any

from haystack.utils import _deserialize_value_with_schema
from haystack.utils.base_serialization import _serialize_with_field_fallback

from haystack_integrations.agent_pack.optimization.tracing.dataclasses import (
    DEFAULT_TRACE_CAPTURE_LIMITS,
    TraceCaptureLimits,
)


def _truncate(value: str, limits: TraceCaptureLimits) -> str:
    """Shorten an oversized string, recording how long it originally was."""
    if len(value) <= limits.max_string_length:
        return value
    return f"{value[: limits.max_string_length]} <truncated: {len(value)} chars total>"


def _apply_plain_limits(data: Any, limits: TraceCaptureLimits) -> Any:
    """Apply the limits to a plain JSON structure that carries no separate schema, such as a `to_dict()` payload."""
    if isinstance(data, str):
        return _truncate(value=data, limits=limits)
    if isinstance(data, dict):
        return {
            key: _apply_plain_limits(data=value, limits=limits)
            for key, value in data.items()
            if key not in limits.dropped_keys
        }
    if isinstance(data, list):
        capped = [_apply_plain_limits(data=item, limits=limits) for item in data[: limits.max_sequence_items]]
        if len(data) > len(capped):
            capped.append({"truncated_items": len(data) - len(capped)})
        return capped
    return data


def _apply_limits(schema: Any, data: Any, limits: TraceCaptureLimits) -> tuple[Any, Any]:
    """
    Apply the limits to a schema-aware value, keeping the schema and the data consistent with each other.

    Only `object` and `array` schemas mirror the data structure, so those are walked in lockstep. Anything else is
    an opaque `to_dict()` payload whose fields have no separate schema entries, and is pruned directly. That is the
    branch document embeddings are dropped in.
    """
    schema_type = schema.get("type") if isinstance(schema, dict) else None

    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties") or {}
        limited_properties: dict[str, Any] = {}
        limited_data: dict[str, Any] = {}
        for key, value in data.items():
            if key in limits.dropped_keys:
                continue
            limited_properties[key], limited_data[key] = _apply_limits(
                schema=properties.get(key, {}), data=value, limits=limits
            )
        return {**schema, "properties": limited_properties}, limited_data

    if schema_type == "array" and isinstance(data, list):
        return _apply_array_limits(schema=schema, data=data, limits=limits)

    if schema_type == "string":
        return schema, _truncate(value=data, limits=limits) if isinstance(data, str) else data

    if schema_type in (None, "null", "boolean", "integer", "number"):
        return schema, data

    # A class-typed payload: `serialized_data` is whatever `to_dict()` produced, with no mirroring schema.
    return schema, _apply_plain_limits(data=data, limits=limits)


def _apply_array_limits(schema: dict[str, Any], data: list[Any], limits: TraceCaptureLimits) -> tuple[Any, Any]:
    """Cap an array, keeping its positional or shared item schema in step with the data that survived."""
    prefix_items = schema.get("prefixItems")
    capped = data[: limits.max_sequence_items]
    limited_schema = dict(schema)

    if prefix_items is not None:
        limited_prefix: list[Any] = []
        limited_data: list[Any] = []
        for index, item in enumerate(capped):
            item_schema, item_data = _apply_limits(
                schema=prefix_items[index] if index < len(prefix_items) else {}, data=item, limits=limits
            )
            limited_prefix.append(item_schema)
            limited_data.append(item_data)
        limited_schema["prefixItems"] = limited_prefix
    else:
        item_schema = schema.get("items") or {}
        limited_data = []
        item_schemas: list[Any] = []
        for item in capped:
            limited_item_schema, item_data = _apply_limits(schema=item_schema, data=item, limits=limits)
            item_schemas.append(limited_item_schema)
            limited_data.append(item_data)
        if item_schemas and any(candidate != item_schemas[0] for candidate in item_schemas):
            # Pruning made the elements structurally different, so the shared `items` schema no longer describes
            # them all. Positional schemas do.
            limited_schema.pop("items", None)
            limited_schema["prefixItems"] = item_schemas
        elif item_schemas:
            limited_schema["items"] = item_schemas[0]

    if len(data) > len(capped):
        # Recorded in the schema rather than appended to the data, so the payload still deserializes into a list of
        # the type it was captured from. Unknown schema keywords are ignored on the way back.
        limited_schema["truncatedItems"] = len(data) - len(capped)
        if "minItems" in limited_schema:
            limited_schema["minItems"] = len(capped)
        if "maxItems" in limited_schema:
            limited_schema["maxItems"] = len(capped)
    return limited_schema, limited_data


def _serialize_trace_value(value: Any, limits: TraceCaptureLimits = DEFAULT_TRACE_CAPTURE_LIMITS) -> dict[str, Any]:
    """
    Convert a Haystack trace value into a bounded, schema-aware payload.

    Uses the same runtime-value serialization Haystack applies to Agent state and pipeline snapshots, so a captured
    tag deserializes back into the objects it was recorded from. Values Haystack cannot serialize are omitted rather
    than raising, because capturing a trace must never break the application being observed.

    :param value: The tag value to convert.
    :param limits: The bounds to apply.
    :returns: A dictionary with `serialization_schema` and `serialized_data` keys.
    """
    payload = _serialize_with_field_fallback(value, description="a captured trace tag")
    schema, data = _apply_limits(
        schema=payload.get("serialization_schema"), data=payload.get("serialized_data"), limits=limits
    )
    return {"serialization_schema": schema, "serialized_data": data}


def deserialize_trace_value(payload: dict[str, Any]) -> Any:
    """
    Restore a captured tag value to the objects it was recorded from.

    :param payload: A payload produced by `_serialize_trace_value`.
    :returns: The deserialized value.
    """
    return _deserialize_value_with_schema(payload)


def span_tag(span: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Read one tag off a captured span record, deserializing it.

    :param span: A span record from `TraceArtifact.traces`.
    :param key: The tag name.
    :param default: Returned when the span carries no such tag.
    :returns: The deserialized tag value.
    """
    payload = (span.get("tags") or {}).get(key)
    if not isinstance(payload, dict) or "serialization_schema" not in payload:
        return default if payload is None else payload
    return deserialize_trace_value(payload=payload)
