# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from math import isfinite
from typing import Any, TypeVar

T = TypeVar("T")
_CODE_LENGTH = 2


def validate_nonblank(value: Any, name: str) -> str:
    """Validate and return a nonblank string."""
    if not isinstance(value, str) or not value.strip():
        msg = f"'{name}' must be a nonblank string."
        raise ValueError(msg)
    return value


def optional_nonblank(value: Any, name: str) -> str | None:
    """Validate an optional string and omit blank values."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"'{name}' must be a string or None."
        raise ValueError(msg)
    return value if value.strip() else None


def validate_int(value: Any, name: str, *, minimum: int) -> int:
    """Validate an integer with an inclusive minimum, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"'{name}' must be an integer greater than or equal to {minimum}."
        raise ValueError(msg)
    if value < minimum:
        msg = f"'{name}' must be greater than or equal to {minimum}."
        raise ValueError(msg)
    return value


def validate_integer(value: Any, name: str) -> int:
    """Validate an integer for an API field with no documented range constraint."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"'{name}' must be an integer."
        raise ValueError(msg)
    return value


def validate_number(value: Any, name: str, *, minimum: float) -> float:
    """Validate a finite numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"'{name}' must be a number greater than {minimum}."
        raise ValueError(msg)
    converted = float(value)
    if converted <= minimum or not isfinite(converted):
        msg = f"'{name}' must be a finite number greater than {minimum}."
        raise ValueError(msg)
    return converted


def validate_bool(value: Any, name: str) -> bool:
    """Validate a strict boolean value."""
    if not isinstance(value, bool):
        msg = f"'{name}' must be a boolean."
        raise ValueError(msg)
    return value


def validate_choice(value: Any, name: str, choices: Iterable[T]) -> T:
    """Validate a value against a finite set of choices."""
    allowed = tuple(choices)
    if value not in allowed:
        msg = f"'{name}' must be one of: {', '.join(repr(choice) for choice in allowed)}."
        raise ValueError(msg)
    return value


def validate_country_code(value: Any, name: str, *, optional: bool = False) -> str | None:
    """Validate a two-letter country, region, or language code."""
    normalized = optional_nonblank(value, name) if optional else validate_nonblank(value, name)
    if normalized is None:
        return None
    if len(normalized) != _CODE_LENGTH or not normalized.isalpha():
        msg = f"'{name}' must be a two-letter alphabetic code."
        raise ValueError(msg)
    return normalized


def validate_dict(value: Any, name: str) -> dict[str, Any]:
    """Validate a dictionary value."""
    if not isinstance(value, dict):
        msg = f"'{name}' must be a dictionary."
        raise ValueError(msg)
    return value


def validate_cookie_list(value: Any) -> list[dict[str, Any]]:
    """Validate a list of browser cookie objects."""
    if not isinstance(value, list) or any(not isinstance(cookie, dict) for cookie in value):
        msg = "'cookies' must be a list of dictionaries."
        raise ValueError(msg)
    return value


def validate_urls(value: Any) -> list[str]:
    """Validate a nonempty list of nonblank URL strings."""
    if not isinstance(value, list) or not value:
        msg = "'urls' must be a nonempty list of nonblank strings."
        raise ValueError(msg)
    urls: list[str] = []
    for index, url in enumerate(value):
        try:
            urls.append(validate_nonblank(url, f"urls[{index}]").strip())
        except ValueError as error:
            msg = "'urls' must be a nonempty list of nonblank strings."
            raise ValueError(msg) from error
    return urls
