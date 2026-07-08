# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict

from haystack_integrations.utils.oauth import OAuthConfigError, SubjectTokenSource, TokenSource


@component
class OAuthTokenResolver:
    """
    Resolves an OAuth access token at pipeline runtime and emits it on the `access_token` output socket.

    The resolver component is a thin wrapper over a pluggable token source that decides *where* the token comes from:
    a standalone OAuth refresh grant (`OAuthRefreshTokenSource`), a per-request token exchange
    (`OAuthTokenExchangeSource`), a static long-lived token (`OAuthStaticTokenSource`), or a custom source you
    provide. A downstream component (for
    example a SharePoint or Google Drive retriever) consumes the token via a normal connection and never knows how
    it was resolved.

    The run input depends on the token source. A source that needs a per-request credential (it sets
    `requires_subject_token = True`, like `OAuthTokenExchangeSource`) makes the resolver declare a **mandatory**
    `subject_token` input — a controller-injected per-request credential (for example an incoming user assertion),
    not chosen by an end user. A config-only source declares no run input, so the resolver is a source node.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
    from haystack_integrations.utils.oauth import OAuthRefreshTokenSource

    resolver = OAuthTokenResolver(
        token_source=OAuthRefreshTokenSource(
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            client_id="aaa-bbb-ccc",
            refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
            scopes=["https://graph.microsoft.com/Files.Read.All", "offline_access"],
        ),
    )
    access_token = resolver.run()["access_token"]
    ```
    """

    def __init__(self, token_source: TokenSource | SubjectTokenSource) -> None:
        """
        Initialize the resolver.

        :param token_source: The strategy that resolves the access token. If it sets `requires_subject_token = True`
            (for example `OAuthTokenExchangeSource`), the resolver declares a mandatory `subject_token` run input;
            otherwise the resolver takes no run input.
        :raises OAuthConfigError: If `token_source` does not implement a token-source protocol.
        """
        if not isinstance(token_source, (TokenSource, SubjectTokenSource)):
            msg = "token_source must implement the TokenSource or SubjectTokenSource protocol."
            raise OAuthConfigError(msg)
        self.token_source = token_source
        if token_source.requires_subject_token:
            # Declare a mandatory run input only when the source needs it (no default => mandatory socket),
            # so a missing per-user credential fails at Pipeline.validate_input rather than at run time.
            component.set_input_type(self, "subject_token", str)

    @staticmethod
    def _subject_token_from(kwargs: dict[str, Any]) -> str:
        """Extract and validate the per-request `subject_token` run input."""
        subject_token = kwargs.get("subject_token")
        if not subject_token:
            msg = "The configured token source requires a non-empty 'subject_token' run input, but none was provided."
            raise OAuthConfigError(msg)
        return subject_token

    @component.output_types(access_token=str)
    def run(self, **kwargs: Any) -> dict[str, str]:
        """
        Resolve an access token and emit it.

        :param kwargs: Carries `subject_token` when the configured source requires it (declared as a mandatory
            input in that case, injected by the application/controller per request). For config-only sources no
            input is declared and `kwargs` is empty.
        :returns: A dictionary with a single `access_token` key containing a bearer token string.
        :raises OAuthConfigError: If the source requires a `subject_token` but it is missing or empty.
        """
        source = self.token_source
        if source.requires_subject_token:
            token = source.resolve(self._subject_token_from(kwargs))
        else:
            token = source.resolve()
        return {"access_token": token}

    @component.output_types(access_token=str)
    async def run_async(self, **kwargs: Any) -> dict[str, str]:
        """
        Asynchronously resolve an access token and emit it.

        :param kwargs: Carries `subject_token` when the configured source requires it.
        :returns: A dictionary with a single `access_token` key containing a bearer token string.
        :raises OAuthConfigError: If the source requires a `subject_token` but it is missing or empty.
        """
        source = self.token_source
        if source.requires_subject_token:
            token = await source.resolve_async(self._subject_token_from(kwargs))
        else:
            token = await source.resolve_async()
        return {"access_token": token}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(self, token_source=self.token_source.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthTokenResolver":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        :raises ImportError: If the serialized `token_source` type cannot be imported.
        """
        return default_from_dict(cls, data)
