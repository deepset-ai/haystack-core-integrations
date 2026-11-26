# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret

from .errors import OAuthConfigError
from .protocols import TokenSource
from .sources import (
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_EXPIRY_BUFFER_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    OAuthClientCredentialsSource,
    OAuthPrivateKeyJWTSource,
    OAuthTokenExchangeSource,
)

DEFAULT_ENTRA_AUTHORITY = "https://login.microsoftonline.com"
# The audience of the token that a blueprint requests for one of its agent identities (global cloud).
DEFAULT_TOKEN_EXCHANGE_AUDIENCE = "api://AzureADTokenExchange"


class EntraAgentOBOTokenSource:
    """
    Resolves access tokens for a Microsoft Entra Agent ID acting on behalf of the signed-in user.

    It implements the
    [agent on-behalf-of flow](https://learn.microsoft.com/en-us/entra/agent-id/agent-on-behalf-of-oauth-flow)
    for agent identities created from an agent identity blueprint. Each resolution takes the user's token as the
    per-request `subject_token` and performs up to two requests against the tenant's token endpoint:

    1. The blueprint authenticates with its own credential and obtains an intermediate token for the agent identity.
       This token is cached until shortly before it expires.
    2. The agent identity exchanges that token and the user's token for an access token to the requested resource.
       The result is cached per user token.

    Configure exactly one credential for the blueprint:

    - `client_secret`: simplest; Microsoft advises against client secrets in production.
    - `certificate` and `private_key`: requires `pip install "oauth-haystack[jwt]"`.
    - `client_assertion_source`: any other config-only token source that yields a client assertion for the blueprint,
      for example the token of a managed identity that is registered as a federated identity credential of the
      blueprint.

    What Microsoft Entra expects, and the errors you get otherwise:

    - The user's token must be issued for the blueprint (its `aud` claim is the blueprint's client ID). A token for
      another resource, such as Microsoft Graph, is rejected with `AADSTS50013`.
    - The agent identity needs the requested delegated permissions. They are granted to it by an administrator,
      either directly or as inheritable permissions on the blueprint, because an agent identity cannot ask the user
      for consent itself. Without them the exchange fails with `AADSTS82014`.

    An access token is valid for a single resource, so use one resolver per downstream service.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
    from haystack_integrations.utils.oauth import EntraAgentOBOTokenSource

    resolver = OAuthTokenResolver(
        token_source=EntraAgentOBOTokenSource(
            tenant_id="<tenant-id>",
            blueprint_client_id="<blueprint-client-id>",
            agent_identity_client_id="<agent-identity-client-id>",
            scopes=["https://graph.microsoft.com/.default"],
            client_secret=Secret.from_env_var("ENTRA_BLUEPRINT_CLIENT_SECRET"),
        ),
    )
    # `user_token` is the signed-in user's access token for the blueprint, injected per request.
    access_token = resolver.run(subject_token=user_token)["access_token"]
    ```

    The same source authenticating the blueprint with a certificate instead of a client secret:
    ```python
    token_source = EntraAgentOBOTokenSource(
        tenant_id="<tenant-id>",
        blueprint_client_id="<blueprint-client-id>",
        agent_identity_client_id="<agent-identity-client-id>",
        scopes=["https://graph.microsoft.com/.default"],
        certificate=Secret.from_env_var("ENTRA_BLUEPRINT_CERTIFICATE"),
        private_key=Secret.from_env_var("ENTRA_BLUEPRINT_PRIVATE_KEY"),
    )
    ```
    """

    requires_subject_token: Literal[True] = True

    def __init__(
        self,
        tenant_id: str,
        blueprint_client_id: str,
        agent_identity_client_id: str,
        scopes: list[str],
        *,
        client_secret: Secret | None = None,
        certificate: Secret | None = None,
        private_key: Secret | None = None,
        client_assertion_source: TokenSource | None = None,
        authority: str = DEFAULT_ENTRA_AUTHORITY,
        token_exchange_audience: str = DEFAULT_TOKEN_EXCHANGE_AUDIENCE,
        expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
        cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the source.

        :param tenant_id: The Microsoft Entra tenant ID.
        :param blueprint_client_id: The client ID of the agent identity blueprint.
        :param agent_identity_client_id: The client ID of the agent identity that acts for the user.
        :param scopes: The scopes of the downstream resource to request, for example
            `["https://graph.microsoft.com/.default"]`. All scopes must belong to the same resource.
        :param client_secret: The blueprint's client secret.
        :param certificate: The blueprint's PEM-encoded X.509 certificate. Must be set together with `private_key`.
        :param private_key: The PEM-encoded, unencrypted private key of `certificate`.
        :param client_assertion_source: A config-only token source that yields a client assertion for the blueprint,
            for credentials not covered by the parameters above.
        :param authority: The Microsoft Entra authority host. Change it for sovereign clouds.
        :param token_exchange_audience: The token-exchange audience of the cloud. The US Government and China clouds
            use `api://AzureADTokenExchangeUSGov` and `api://AzureADTokenExchangeChina`.
        :param expiry_buffer_seconds: Request a new token this many seconds before a cached one expires.
        :param cache_max_size: The maximum number of per-user access tokens to keep in the in-memory cache. The
            least-recently-used entry is evicted when the cache is full.
        :param timeout: The timeout, in seconds, for each request to the token endpoint.
        :raises ImportError: If a certificate is configured and the `jwt` extra is not installed.
        :raises OAuthConfigError: If a required value is empty, or if the blueprint credential is missing or
            ambiguous.
        """
        for name, value in (
            ("tenant_id", tenant_id),
            ("blueprint_client_id", blueprint_client_id),
            ("agent_identity_client_id", agent_identity_client_id),
            ("authority", authority),
            ("token_exchange_audience", token_exchange_audience),
        ):
            if not value:
                msg = f"{name} must be a non-empty string."
                raise OAuthConfigError(msg)
        if not scopes:
            msg = "scopes must contain at least one scope."
            raise OAuthConfigError(msg)
        if (certificate is None) != (private_key is None):
            msg = "certificate and private_key must be set together."
            raise OAuthConfigError(msg)
        credentials = (client_secret, certificate, client_assertion_source)
        if sum(credential is not None for credential in credentials) != 1:
            msg = (
                "Exactly one blueprint credential is required: client_secret, certificate with private_key, or "
                "client_assertion_source."
            )
            raise OAuthConfigError(msg)

        self.tenant_id = tenant_id
        self.blueprint_client_id = blueprint_client_id
        self.agent_identity_client_id = agent_identity_client_id
        self.scopes = list(scopes)
        self.client_secret = client_secret
        self.certificate = certificate
        self.private_key = private_key
        self.client_assertion_source = client_assertion_source
        self.authority = authority
        self.token_exchange_audience = token_exchange_audience
        self.expiry_buffer_seconds = expiry_buffer_seconds
        self.cache_max_size = cache_max_size
        self.timeout = timeout

        token_url = f"{authority.rstrip('/')}/{tenant_id}/oauth2/v2.0/token"
        token_exchange_scope = f"{token_exchange_audience}/.default"
        blueprint_assertion_source = client_assertion_source
        if certificate is not None and private_key is not None:
            # Microsoft Entra requires PS256 and identifies the certificate by its SHA-256 thumbprint.
            blueprint_assertion_source = OAuthPrivateKeyJWTSource(
                client_id=blueprint_client_id,
                audience=token_url,
                private_key=private_key,
                certificate=certificate,
                algorithm="PS256",
            )

        self._source = OAuthTokenExchangeSource(
            token_url=token_url,
            client_id=agent_identity_client_id,
            # The blueprint's token for the agent identity authenticates the agent identity's exchange.
            client_assertion_source=OAuthClientCredentialsSource(
                token_url=token_url,
                client_id=blueprint_client_id,
                client_secret=client_secret,
                client_assertion_source=blueprint_assertion_source,
                scopes=[token_exchange_scope],
                extra_token_params={"fmi_path": agent_identity_client_id},
                expiry_buffer_seconds=expiry_buffer_seconds,
                timeout=timeout,
            ),
            grant_type="urn:ietf:params:oauth:grant-type:jwt-bearer",
            subject_token_param="assertion",
            scopes=scopes,
            extra_token_params={"requested_token_use": "on_behalf_of"},
            expiry_buffer_seconds=expiry_buffer_seconds,
            cache_max_size=cache_max_size,
            timeout=timeout,
        )

    def resolve(self, subject_token: str) -> str:
        """
        Exchange the user's token for an access token to the configured resource (cached per user token).

        :param subject_token: The signed-in user's access token, issued for the blueprint.
        :returns: A valid bearer access token for the resource, acting on behalf of the user.
        :raises OAuthConfigError: If `subject_token` is empty.
        :raises TokenRefreshError: If the blueprint credential cannot be used or Microsoft Entra rejects one of the
            two requests. The message carries Entra's error response, including its `AADSTS` code.
        """
        return self._source.resolve(subject_token)

    async def resolve_async(self, subject_token: str) -> str:
        """
        Asynchronous counterpart of `resolve`.

        :param subject_token: The signed-in user's access token, issued for the blueprint.
        :returns: A valid bearer access token for the resource, acting on behalf of the user.
        :raises OAuthConfigError: If `subject_token` is empty.
        :raises TokenRefreshError: If the blueprint credential cannot be used or Microsoft Entra rejects one of the
            two requests. The message carries Entra's error response, including its `AADSTS` code.
        """
        return await self._source.resolve_async(subject_token)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(
            self,
            tenant_id=self.tenant_id,
            blueprint_client_id=self.blueprint_client_id,
            agent_identity_client_id=self.agent_identity_client_id,
            scopes=self.scopes,
            client_secret=self.client_secret,
            certificate=self.certificate,
            private_key=self.private_key,
            client_assertion_source=self.client_assertion_source.to_dict() if self.client_assertion_source else None,
            authority=self.authority,
            token_exchange_audience=self.token_exchange_audience,
            expiry_buffer_seconds=self.expiry_buffer_seconds,
            cache_max_size=self.cache_max_size,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntraAgentOBOTokenSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)
