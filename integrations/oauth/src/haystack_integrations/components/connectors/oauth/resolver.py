# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from haystack import component, default_from_dict, default_to_dict

from .errors import OAuthConfigError
from .sources import SubjectTokenSource, TokenSource


@component
class OAuthResolver:
    """
    Resolves an OAuth access token at pipeline runtime and emits it on the `access_token` output socket.

    The resolver is a thin graph node over a pluggable token source that decides *where* the token comes from:
    a standalone OAuth refresh grant (`RefreshTokenSource`), a per-request token exchange (`TokenExchangeSource`),
    a static long-lived token (`StaticTokenSource`), or a custom source you provide. A downstream component (for
    example a SharePoint or Google Drive retriever) consumes the token via a normal connection and never knows how
    it was resolved.

    The run input depends on the token source. A source that needs a per-request credential (it sets
    `requires_subject_token = True`, like `TokenExchangeSource`) makes the resolver declare a **mandatory**
    `subject_token` input — a controller-injected per-request credential (for example an incoming user assertion),
    not chosen by an end user. A config-only source declares no run input, so the resolver is a source node.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.connectors.oauth import OAuthResolver, RefreshTokenSource

    resolver = OAuthResolver(
        token_source=RefreshTokenSource(
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            client_id="aaa-bbb-ccc",
            refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
            scopes=["https://graph.microsoft.com/Files.Read.All", "offline_access"],
        ),
    )
    access_token = resolver.run()["access_token"]
    ```
    """

    def __init__(self, token_source: "TokenSource | SubjectTokenSource") -> None:
        """
        Initialize the resolver.

        :param token_source: The strategy that resolves the access token. If it sets `requires_subject_token = True`
            (for example `TokenExchangeSource`), the resolver declares a mandatory `subject_token` run input;
            otherwise the resolver takes no run input.
        :raises OAuthConfigError: If `token_source` does not implement a token-source protocol.
        """
        if not isinstance(token_source, (TokenSource, SubjectTokenSource)):
            msg = "token_source must implement the TokenSource or SubjectTokenSource protocol."
            raise OAuthConfigError(msg)
        self.token_source = token_source
        self._requires_subject_token = bool(getattr(token_source, "requires_subject_token", False))
        if self._requires_subject_token:
            # Declare a mandatory run input only when the source needs it (no default => mandatory socket),
            # so a missing per-user credential fails at Pipeline.validate_input rather than at run time.
            component.set_input_type(self, "subject_token", str)

    @component.output_types(access_token=str)
    def run(self, **kwargs: Any) -> dict[str, str]:
        """
        Resolve an access token and emit it.

        :param kwargs: Carries `subject_token` when the configured source requires it (declared as a mandatory
            input in that case, injected by the application/controller per request). For config-only sources no
            input is declared and `kwargs` is empty.
        :returns: A dictionary with a single `access_token` key containing a bearer token string.
        """
        if self._requires_subject_token:
            token = cast("SubjectTokenSource", self.token_source).resolve(kwargs.get("subject_token", ""))
        else:
            token = cast("TokenSource", self.token_source).resolve()
        return {"access_token": token}

    @component.output_types(access_token=str)
    async def run_async(self, **kwargs: Any) -> dict[str, str]:
        """
        Asynchronously resolve an access token and emit it.

        :param kwargs: Carries `subject_token` when the configured source requires it.
        :returns: A dictionary with a single `access_token` key containing a bearer token string.
        """
        if self._requires_subject_token:
            token = await cast("SubjectTokenSource", self.token_source).resolve_async(kwargs.get("subject_token", ""))
        else:
            token = await cast("TokenSource", self.token_source).resolve_async()
        return {"access_token": token}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(self, token_source=self.token_source.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthResolver":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
