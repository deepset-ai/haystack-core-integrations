# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install plivo'") as plivo_import:
    import plivo

logger = logging.getLogger(__name__)


class PlivoClient:
    """
    Holds Plivo account credentials and lazily builds a shared ``plivo.RestClient``.

    Instantiate this class once and pass it to one or more Plivo tool classes
    (:class:`SendSMSTool`, :class:`SendVerificationTool`, :class:`ValidateVerificationTool`,
    :class:`LookupNumberTool`, :class:`MakeCallTool`) so they all authenticate with the
    same account. Credentials are read from ``PLIVO_AUTH_ID`` and ``PLIVO_AUTH_TOKEN``
    by default. The optional ``sender`` is the Plivo number (or approved sender id) used
    as the SMS ``src`` and the outbound-call caller id.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    from haystack_integrations.tools.plivo import PlivoClient, SendSMSTool

    client = PlivoClient(sender="+14150000000")
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=[SendSMSTool(client=client)],
    )
    ```
    """

    def __init__(
        self,
        auth_id: Secret = Secret.from_env_var("PLIVO_AUTH_ID", strict=True),
        auth_token: Secret = Secret.from_env_var("PLIVO_AUTH_TOKEN", strict=True),
        sender: str | None = None,
    ) -> None:
        """
        Create a PlivoClient.

        :param auth_id: Plivo Auth ID. Defaults to ``Secret.from_env_var("PLIVO_AUTH_ID")``.
        :param auth_token: Plivo Auth Token. Defaults to ``Secret.from_env_var("PLIVO_AUTH_TOKEN")``.
        :param sender: Plivo number or approved sender id in E.164, used as the SMS ``src``
            and the outbound-call caller id. Required only by the tools that send from an
            account number (SMS and calls).
        """
        self.auth_id = auth_id
        self.auth_token = auth_token
        self.sender = sender
        self._client: Any = None

    def warm_up(self) -> None:
        """
        Build the underlying ``plivo.RestClient``.

        Idempotent -- calling it multiple times reuses the first client.

        :raises RuntimeError: If the client cannot be created.
        """
        if self._client is not None:
            return

        plivo_import.check()
        try:
            self._client = plivo.RestClient(
                auth_id=self.auth_id.resolve_value(),
                auth_token=self.auth_token.resolve_value(),
            )
            logger.info("Plivo client initialized")
        except Exception as e:
            msg = f"Failed to initialize Plivo client: {e}"
            raise RuntimeError(msg) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the client configuration to a dictionary.

        :returns: Dictionary containing the serialised configuration.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "auth_id": self.auth_id.to_dict(),
                "auth_token": self.auth_token.to_dict(),
                "sender": self.sender,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlivoClient":
        """
        Deserialize a :class:`PlivoClient` from a dictionary.

        :param data: Dictionary created by :meth:`to_dict`.
        :returns: A :class:`PlivoClient` instance ready to be warmed up.
        """
        inner = data["data"]
        deserialize_secrets_inplace(inner, keys=["auth_id", "auth_token"])
        return cls(
            auth_id=inner["auth_id"],
            auth_token=inner["auth_token"],
            sender=inner.get("sender"),
        )

    def _require_client(self) -> Any:
        """Return the active RestClient, building it on first use."""
        if self._client is None:
            self.warm_up()
        return self._client

    def _require_sender(self) -> str:
        """Return the configured sender or raise a helpful error."""
        if not self.sender:
            msg = (
                "No sender configured. Pass 'sender' (a Plivo number or approved sender id "
                "in E.164) to PlivoClient to send SMS or place calls."
            )
            raise RuntimeError(msg)
        return self.sender
