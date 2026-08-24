# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Resolving which model a chat generator is configured with."""

from collections.abc import Mapping
from typing import Any

from haystack.core.serialization import component_to_dict

# Chat generators disagree on where the model identifier lives: most use `model`, Azure deployments use
# `azure_deployment`, a few expose `model_name`, and the Hugging Face API generators nest it inside `api_params`.
# Probing all of them keeps configuration-time validation from rejecting a perfectly approved Azure or HF harness.
_MODEL_KEYS = ("model", "azure_deployment", "model_name")
_NESTED_MODEL_CONTAINERS = ("api_params",)
_NESTED_MODEL_KEYS = ("model", "repo_id")


def _model_id_path(init_parameters: Mapping[str, Any]) -> tuple[str, ...] | None:
    """
    Return the key path holding a model identifier inside a component's init parameters.

    :param init_parameters: The serialized init parameters to inspect.
    :returns: The key path to the identifier, or None if the component declares none.
    """
    for key in _MODEL_KEYS:
        if isinstance(init_parameters.get(key), str):
            return (key,)
    for container in _NESTED_MODEL_CONTAINERS:
        nested = init_parameters.get(container)
        if isinstance(nested, Mapping):
            for key in _NESTED_MODEL_KEYS:
                if isinstance(nested.get(key), str):
                    return (container, key)
    return None


def _init_parameters_of(serialized_component: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return a serialized component's init parameters, tolerating both container key conventions.

    :param serialized_component: A serialized component or tool.
    :returns: Its init parameters, or an empty mapping when it declares none.
    """
    parameters = serialized_component.get("init_parameters") or serialized_component.get("data") or {}
    return parameters if isinstance(parameters, Mapping) else {}


def serialized_model_id(serialized_component: Mapping[str, Any]) -> str | None:
    """
    Return the model identifier configured on a serialized component.

    :param serialized_component: A serialized component, as produced by `component_to_dict`.
    :returns: The configured model identifier, or None if the component declares none.
    """
    init_parameters = _init_parameters_of(serialized_component=serialized_component)
    path = _model_id_path(init_parameters=init_parameters)
    if path is None:
        return None
    value: Any = init_parameters
    for key in path:
        value = value[key]
    return value if isinstance(value, str) else None


def generator_model_id(generator: Any) -> str | None:
    """
    Return the model identifier of a live chat generator.

    Attributes are probed first because they are cheap and because generators built for tests may not serialize
    their model at all. Serialization is the fallback for generators that only expose the identifier that way.

    :param generator: The chat generator to inspect.
    :returns: The configured model identifier, or None if it cannot be determined.
    """
    for key in _MODEL_KEYS:
        value = getattr(generator, key, None)
        if isinstance(value, str):
            return value
    for container in _NESTED_MODEL_CONTAINERS:
        nested = getattr(generator, container, None)
        if isinstance(nested, Mapping):
            for key in _NESTED_MODEL_KEYS:
                if isinstance(nested.get(key), str):
                    return str(nested[key])
    try:
        serialized = component_to_dict(obj=generator, name="chat_generator")
    except Exception:
        return None
    return serialized_model_id(serialized_component=serialized)
