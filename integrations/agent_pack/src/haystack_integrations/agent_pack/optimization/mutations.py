# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Structured operations that can change any serialized Agent configuration."""

import hashlib
from copy import deepcopy
from typing import Any, Literal, Self, TypeAlias

from haystack.components.agents import Agent
from pydantic import BaseModel, ConfigDict, model_validator

ScalarValue: TypeAlias = str | int | float | bool | None


class MutationOperation(BaseModel):
    """One flat, strict-schema-compatible operation over an Agent configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    op: Literal["set", "create_object", "create_array", "remove", "copy"]
    path: str
    value: ScalarValue = None
    from_path: str | None = None

    @model_validator(mode="after")
    def validate_arguments(self) -> Self:
        """Require copy sources and reject fields that the selected operation does not use."""
        if self.op == "copy" and self.from_path is None:
            msg = "A copy operation requires from_path."
            raise ValueError(msg)
        if self.op != "copy" and self.from_path is not None:
            msg = f"A {self.op} operation must set from_path to null."
            raise ValueError(msg)
        if self.op != "set" and self.value is not None:
            msg = f"A {self.op} operation must set value to null."
            raise ValueError(msg)
        return self


class AgentMutation(BaseModel):
    """One candidate expressed as ordered edits to the reference Agent configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operations: tuple[MutationOperation, ...]


class OptimizerDecision(BaseModel):
    """The optimizer's next candidate mutation, or ``null`` when experimentation should stop."""

    model_config = ConfigDict(extra="forbid")
    mutation: AgentMutation | None


def _segments(path: str) -> list[str]:
    """Decode an RFC 6901 JSON Pointer into path segments."""
    if path == "":
        return []
    if not path.startswith("/"):
        msg = f"Configuration path {path!r} must be an RFC 6901 JSON Pointer."
        raise ValueError(msg)
    return [segment.replace("~1", "/").replace("~0", "~") for segment in path[1:].split("/")]


def _array_index(segment: str, length: int, allow_append: bool) -> int:
    """Resolve one JSON Pointer array segment to a checked integer index."""
    if segment == "-" and allow_append:
        return length
    try:
        index = int(segment)
    except ValueError as error:
        msg = f"Array path segment {segment!r} is not an index."
        raise ValueError(msg) from error
    maximum = length if allow_append else length - 1
    if index < 0 or index > maximum:
        msg = f"Array index {index} is outside the allowed range 0..{maximum}."
        raise ValueError(msg)
    return index


def _read(root: Any, path: str) -> Any:
    """Read one value from a JSON-compatible tree by JSON Pointer."""
    current = root
    for segment in _segments(path=path):
        if isinstance(current, dict):
            if segment not in current:
                msg = f"Configuration path {path!r} does not exist."
                raise ValueError(msg)
            current = current[segment]
        elif isinstance(current, list):
            current = current[_array_index(segment=segment, length=len(current), allow_append=False)]
        else:
            msg = f"Configuration path {path!r} traverses a scalar value."
            raise ValueError(msg)
    return current


def _parent(root: Any, segments: list[str], path: str) -> Any:
    """Traverse already-decoded path segments to the parent of a target value."""
    current = root
    for segment in segments[:-1]:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        elif isinstance(current, list):
            current = current[_array_index(segment=segment, length=len(current), allow_append=False)]
        else:
            msg = f"Configuration path {path!r} does not have an existing container parent."
            raise ValueError(msg)
    return current


def _write(root: Any, path: str, value: Any) -> Any:
    """Add or replace one value in a JSON-compatible tree by JSON Pointer."""
    segments = _segments(path=path)
    if not segments:
        return value
    parent = _parent(root=root, segments=segments, path=path)
    final = segments[-1]
    if isinstance(parent, dict):
        parent[final] = value
    elif isinstance(parent, list):
        index = _array_index(segment=final, length=len(parent), allow_append=True)
        if index == len(parent):
            parent.append(value)
        else:
            parent[index] = value
    else:
        msg = f"Configuration path {path!r} has a scalar parent."
        raise ValueError(msg)
    return root


def _remove(root: Any, path: str) -> Any:
    """Remove one value from a JSON-compatible tree by JSON Pointer."""
    segments = _segments(path=path)
    if not segments:
        msg = "The root Agent configuration cannot be removed."
        raise ValueError(msg)
    parent = _parent(root=root, segments=segments, path=path)
    final = segments[-1]
    if isinstance(parent, dict):
        if final not in parent:
            msg = f"Configuration path {path!r} does not exist."
            raise ValueError(msg)
        del parent[final]
    elif isinstance(parent, list):
        parent.pop(_array_index(segment=final, length=len(parent), allow_append=False))
    else:
        msg = f"Configuration path {path!r} has a scalar parent."
        raise ValueError(msg)
    return root


def apply_mutation(serialized_agent: dict[str, Any], mutation: AgentMutation) -> dict[str, Any]:
    """Apply ordered structured operations to an independent Agent configuration copy."""
    changed: Any = deepcopy(serialized_agent)
    for operation in mutation.operations:
        if operation.op == "set":
            changed = _write(root=changed, path=operation.path, value=operation.value)
        elif operation.op == "create_object":
            changed = _write(root=changed, path=operation.path, value={})
        elif operation.op == "create_array":
            changed = _write(root=changed, path=operation.path, value=[])
        elif operation.op == "remove":
            changed = _remove(root=changed, path=operation.path)
        else:
            if operation.from_path is None:
                msg = "A copy operation requires from_path."
                raise ValueError(msg)
            copied = deepcopy(_read(root=changed, path=operation.from_path))
            changed = _write(root=changed, path=operation.path, value=copied)
    if not isinstance(changed, dict):
        msg = "A mutated Agent configuration must remain an object."
        raise ValueError(msg)
    return changed


def materialize_mutation(reference: Agent, mutation: AgentMutation) -> tuple[Agent, dict[str, Any]]:
    """Apply a mutation and deserialize the resulting candidate Agent."""
    try:
        serialized_reference = reference.to_dict()
    except Exception as error:
        msg = f"{type(reference).__name__} cannot be serialized and optimized: {error}"
        raise ValueError(msg) from error
    serialized_candidate = apply_mutation(serialized_agent=serialized_reference, mutation=mutation)
    try:
        candidate = type(reference).from_dict(data=deepcopy(serialized_candidate))
    except Exception as error:
        msg = f"The mutated Agent configuration could not be rebuilt: {error}"
        raise ValueError(msg) from error
    return candidate, serialized_candidate


def mutation_fingerprint(mutation: AgentMutation) -> str:
    """Fingerprint an invalid mutation that has no resulting Agent configuration."""
    payload = mutation.model_dump_json()
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "AgentMutation",
    "MutationOperation",
    "OptimizerDecision",
    "apply_mutation",
    "materialize_mutation",
    "mutation_fingerprint",
]
