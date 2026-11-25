# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .entra import EntraAgentOBOTokenSource
from .errors import OAuthConfigError, OAuthError, TokenRefreshError
from .protocols import SubjectTokenSource, TokenSource
from .sources import (
    OAuthClientCredentialsSource,
    OAuthPrivateKeyJWTSource,
    OAuthRefreshTokenSource,
    OAuthStaticTokenSource,
    OAuthTokenExchangeSource,
)

__all__ = [
    "EntraAgentOBOTokenSource",
    "OAuthClientCredentialsSource",
    "OAuthConfigError",
    "OAuthError",
    "OAuthPrivateKeyJWTSource",
    "OAuthRefreshTokenSource",
    "OAuthStaticTokenSource",
    "OAuthTokenExchangeSource",
    "SubjectTokenSource",
    "TokenRefreshError",
    "TokenSource",
]
