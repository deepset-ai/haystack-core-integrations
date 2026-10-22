# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
from urllib.parse import quote

_REPO_PART_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def _encode_segments(value: str, kind: str) -> str:
    # `quote` never encodes ".", so dot segments must be rejected before encoding: requests/urllib3 resolve them
    # and would move the request to a different endpoint.
    segments = value.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        msg = f"Invalid {kind} '{value}': empty, '.' and '..' segments are not allowed"
        raise ValueError(msg)
    return "/".join(quote(segment, safe="") for segment in segments)


def encode_path(path: str, *, allow_empty: bool = False) -> str:
    """
    Percent-encode a repository file path for use in a GitHub REST API URL path.

    Leading and trailing slashes are stripped. Every other character, including `?`, `#` and `%`, is encoded so that it
    stays part of the file name.

    :param path: Path of a file or directory inside the repository.
    :param allow_empty: Whether an empty path (the repository root) is accepted.
    :returns: The encoded path.
    :raises ValueError: If the path contains empty, `.` or `..` segments, or is empty and `allow_empty` is False.
    """
    stripped = path.strip("/")
    if not stripped and allow_empty:
        return ""
    return _encode_segments(stripped, "path")


def encode_ref(ref: str) -> str:
    """
    Percent-encode a branch name for use in a GitHub REST API URL path.

    Each `/`-separated segment is encoded separately, so branch names such as `feature/x` keep their slashes.

    :param ref: Branch name.
    :returns: The encoded branch name.
    :raises ValueError: If the branch name is empty or contains empty, `.` or `..` segments.
    """
    return _encode_segments(ref, "branch")


def validate_repo_part(name: str) -> None:
    """
    Check that a repository owner or name is safe to use in a GitHub REST API URL path.

    :param name: Repository owner or repository name.
    :raises ValueError: If the name contains characters other than letters, digits, `.`, `_` and `-`, or is `.` or
        `..`.
    """
    if not _REPO_PART_PATTERN.fullmatch(name) or name in (".", ".."):
        msg = f"Invalid repository owner or name '{name}'"
        raise ValueError(msg)
