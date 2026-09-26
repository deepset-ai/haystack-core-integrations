# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared compatibility helpers for the ``tokenizer_kwargs`` to ``processor_kwargs`` migration."""

from typing import Any

from haystack import logging

logger = logging.getLogger(__name__)

_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."


def _resolve_processor_kwargs(
    *,
    tokenizer_kwargs: dict[str, Any] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    warn: bool = True,
) -> dict[str, Any] | None:
    """Select the effective loader kwargs, preferring the canonical name without merging or mutating inputs."""
    if warn and tokenizer_kwargs is not None:
        message = _DEPRECATED_MESSAGE
        if processor_kwargs is not None:
            message += _CONFLICT_SUFFIX
        logger.warning(message)
    if processor_kwargs is not None:
        return processor_kwargs
    return tokenizer_kwargs


def _normalize_processor_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Copy a serialized mapping so ``init_parameters`` carries only the canonical kwargs name."""
    normalized = dict(data)
    init_parameters = normalized.get("init_parameters")
    if not isinstance(init_parameters, dict):
        return normalized
    parameters = dict(init_parameters)
    normalized["init_parameters"] = parameters
    if "tokenizer_kwargs" not in parameters and "processor_kwargs" not in parameters:
        return normalized
    legacy = parameters.pop("tokenizer_kwargs", None)
    canonical = parameters.get("processor_kwargs")
    parameters["processor_kwargs"] = _resolve_processor_kwargs(
        tokenizer_kwargs=legacy, processor_kwargs=canonical, warn=legacy is not None and canonical is not None
    )
    return normalized
