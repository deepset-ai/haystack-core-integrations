# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the candidate recipes."""

from copy import deepcopy
from typing import Any

from haystack.components.agents import Agent

_LIST_NAME_KEYS = ("data", "init_parameters")


def _clone_agent(reference: Agent, **overrides: Any) -> Agent:
    """
    Clone an Agent without sharing the mutable containers `Agent.clone` copies by reference.

    `Agent.clone` re-reads each init parameter off the instance, so a clone shares the reference's `hooks` dict,
    `state_schema` dict, and `tools` list. Registering a hook or appending a tool on a candidate would then mutate
    the reference harness. Fresh containers are substituted here; the hook and tool objects themselves are still
    shared, so a stateful hook instance remains shared between reference and candidate.

    :param reference: The Agent to clone.
    :param overrides: Init parameters to replace on the clone.
    :returns: The new Agent.
    """
    if "hooks" not in overrides and reference.hooks:
        overrides["hooks"] = {point: list(hooks) for point, hooks in reference.hooks.items()}
    if "state_schema" not in overrides:
        schema = getattr(reference, "state_schema", None)
        if isinstance(schema, dict):
            overrides["state_schema"] = dict(schema)
    if "tools" not in overrides and isinstance(reference.tools, list):
        overrides["tools"] = list(reference.tools)
    return reference.clone(**overrides)


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
    if isinstance(current, list):
        msg = f"Patch path {path!r} ends at a list, which cannot be assigned to by name."
        raise ValueError(msg)
    if not isinstance(current, dict):
        msg = f"Patch path {path!r} runs into a value that cannot be traversed."
        raise ValueError(msg)
    return current, segments[-1]


def _patched_agent(reference: Agent, patch: dict[str, Any]) -> Agent:
    """
    Rebuild an Agent from its serialized form with the patch applied.

    Going through `to_dict`/`from_dict` is what lets one mechanism reach every init parameter of every component,
    and it gives a candidate that shares nothing mutable with the reference. Two things follow from that. A harness
    holding locally defined function tools or closures does not serialize, so it cannot be patched. And rebuilding
    imports every component by name, which Haystack gates behind a module allowlist covering its own packages, so a
    harness containing components from another package needs that package allowed first, through
    `haystack.core.serialization_security.allow_deserialization_module` or the
    `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

    :param reference: The harness to patch, left untouched.
    :param patch: Dotted paths mapped to the values to set, relative to the Agent's init parameters.
    :returns: The patched Agent.
    :raises ValueError: If the harness does not serialize, or a path cannot be resolved.
    """
    try:
        data = reference.to_dict()
    except Exception as error:
        msg = (
            f"{type(reference).__name__} does not serialize, so it cannot be patched: {error}. Harnesses holding "
            "locally defined function tools or closures have to be changed through a recipe that works on the live "
            "objects instead."
        )
        raise ValueError(msg) from error

    data = deepcopy(data)
    init_parameters = data.get("init_parameters")
    if not isinstance(init_parameters, dict):
        msg = f"{type(reference).__name__} serialized without init parameters, so it cannot be patched."
        raise ValueError(msg)
    for path, value in patch.items():
        container, key = _resolve_patch_target(data=init_parameters, path=path)
        container[key] = value
    try:
        return type(reference).from_dict(data)
    except Exception as error:
        msg = (
            f"The patched harness could not be rebuilt: {error}. Rebuilding imports every component by name, which "
            "Haystack gates behind a module allowlist; allow the package holding your components with "
            "`allow_deserialization_module` or the HAYSTACK_DESERIALIZATION_ALLOWLIST environment variable."
        )
        raise ValueError(msg) from error
