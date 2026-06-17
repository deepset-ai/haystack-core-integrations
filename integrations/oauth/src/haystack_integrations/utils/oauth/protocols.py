# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class TokenSource(Protocol):
    """
    A token source that resolves an access token with no per-request input (a config-only source).

    Implemented by sources whose credential is fixed at construction time — e.g. `RefreshTokenSource` and
    `StaticTokenSource`. Such sources set the class attribute `requires_subject_token = False`, and `OAuthResolver`
    runs them as source nodes (no run input).
    """

    requires_subject_token: Literal[False] = False

    def resolve(self) -> str:
        """Return a valid access token."""
        ...

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenSource":
        """Deserialize the source from a dictionary."""
        ...


@runtime_checkable
class SubjectTokenSource(Protocol):
    """
    A token source that resolves an access token by exchanging a per-request subject token.

    The `subject_token` is a controller-injected per-request credential (for example an incoming user assertion),
    not chosen by an end user. Implemented by `TokenExchangeSource`. Such sources set the class attribute
    `requires_subject_token = True`, which makes `OAuthResolver` declare a mandatory `subject_token` run input.
    """

    requires_subject_token: Literal[True] = True

    def resolve(self, subject_token: str) -> str:
        """Return a valid access token for the per-request `subject_token`."""
        ...

    async def resolve_async(self, subject_token: str) -> str:
        """Asynchronous counterpart of `resolve`."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectTokenSource":
        """Deserialize the source from a dictionary."""
        ...
