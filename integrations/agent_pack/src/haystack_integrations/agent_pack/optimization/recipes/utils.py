# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Rebuilding a harness from its serialized form with a patch applied."""

from copy import deepcopy
from typing import Any

from haystack.components.agents import Agent

_LIST_NAME_KEYS = ("data", "init_parameters")


def _serialized_harness(reference: Agent) -> dict[str, Any]:
    """
    Serialize a harness so a patch can be applied to it.

    :param reference: The harness to serialize, left untouched.
    :returns: A deep copy of the serialized harness, safe to modify.
    :raises ValueError: If the harness does not serialize.
    """
    try:
        data = reference.to_dict()
    except Exception as error:
        msg = (
            f"{type(reference).__name__} does not serialize, so it cannot be changed: {error}. Harnesses holding "
            "locally defined function tools or closures cannot be used with an optimization experiment."
        )
        raise ValueError(msg) from error
    if not isinstance(data.get("init_parameters"), dict):
        msg = f"{type(reference).__name__} serialized without init parameters, so it cannot be changed."
        raise ValueError(msg)
    return deepcopy(data)


def _named_element(elements: list[Any], name: str) -> dict[str, Any] | None:
    """
    Find the element of a serialized list that carries this name, such as one tool among an Agent's tools.

    Returns the container the name was found in rather than the element wrapping it, so a path reads
    `tools.search_documents.component...` and does not have to name the `data` key serialization puts fields under.
    """
    for element in elements:
        if not isinstance(element, dict):
            continue
        for key in _LIST_NAME_KEYS:
            container = element.get(key)
            if isinstance(container, dict) and container.get("name") == name:
                return container
    return None


def _resolve_patch_target(data: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    """
    Walk a dotted path through a serialized harness and return the container to write into, plus the final key.

    A segment addressing a list selects the element carrying that name, which is how a tool is reached by name
    rather than by position. Missing intermediate dictionaries are created, so a nested generation parameter can be
    set on a component that declares none.

    :param data: The serialized structure to walk, an Agent's init parameters.
    :param path: The dotted path to resolve.
    :returns: The container holding the final key, and the final key.
    :raises ValueError: If a segment names a list element that does not exist, or a path runs into a non-container.
    """
    segments = path.split(".")
    current: Any = data
    for index, segment in enumerate(segments[:-1]):
        if isinstance(current, list):
            element = _named_element(elements=current, name=segment)
            if element is None:
                msg = f"Patch path {path!r} names {segment!r}, which is not in the harness."
                raise ValueError(msg)
            current = element
            continue
        if not isinstance(current, dict):
            reached = ".".join(segments[:index])
            msg = f"Patch path {path!r} runs into a value at {reached!r} that cannot be traversed."
            raise ValueError(msg)
        current = current.setdefault(segment, {})
    if not isinstance(current, dict):
        msg = f"Patch path {path!r} runs into a value that cannot be traversed."
        raise ValueError(msg)
    return current, segments[-1]


def _rebuild_harness(*, reference: Agent, data: dict[str, Any], patch: dict[str, Any]) -> Agent:
    """
    Apply a patch to a serialized harness and rebuild it.

    Going through `to_dict`/`from_dict` is the one mechanism every transformation uses. It reaches any init
    parameter of any component, and it gives a candidate that shares nothing with the reference, so no candidate can
    mutate the harness it is being compared against. Rebuilding imports every component by name, which Haystack
    gates behind a module allowlist covering its own packages: a harness containing components from another package
    needs that package allowed first, through
    `haystack.core.serialization_security.allow_deserialization_module` or the
    `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

    :param reference: The harness the serialized form came from, used for its type.
    :param data: The serialized harness, modified in place.
    :param patch: Dotted paths mapped to the values to set, relative to the Agent's init parameters.
    :returns: The rebuilt Agent.
    :raises ValueError: If a path does not resolve, or the patched harness cannot be rebuilt.
    """
    init_parameters = data["init_parameters"]
    for path, value in patch.items():
        container, key = _resolve_patch_target(data=init_parameters, path=path)
        container[key] = value
    try:
        return type(reference).from_dict(data)
    except Exception as error:
        msg = (
            f"The changed harness could not be rebuilt: {error}. Rebuilding imports every component by name, which "
            "Haystack gates behind a module allowlist; allow the package holding your components with "
            "`allow_deserialization_module` or the HAYSTACK_DESERIALIZATION_ALLOWLIST environment variable."
        )
        raise ValueError(msg) from error


def _patched_agent(*, reference: Agent, patch: dict[str, Any]) -> Agent:
    """
    Rebuild an Agent from its serialized form with the patch applied.

    :param reference: The harness to change, left untouched.
    :param patch: Dotted paths mapped to the values to set, relative to the Agent's init parameters.
    :returns: The changed Agent.
    :raises ValueError: If the harness does not serialize, or a path does not resolve.
    """
    return _rebuild_harness(reference=reference, data=_serialized_harness(reference=reference), patch=patch)
