# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class OAuthError(Exception):
    """Base class for errors raised by the OAuth integration."""


class OAuthConfigError(OAuthError):
    """Raised when an OAuth component or token source is misconfigured."""


class TokenRefreshError(OAuthError):
    """Raised when a token cannot be resolved or refreshed at the identity provider."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
