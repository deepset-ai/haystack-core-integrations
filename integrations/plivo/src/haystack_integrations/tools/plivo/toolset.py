# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Toolset
from haystack.utils import Secret

from haystack_integrations.tools.plivo.call_tool import MakeCallTool
from haystack_integrations.tools.plivo.client import PlivoClient
from haystack_integrations.tools.plivo.lookup_tool import LookupNumberTool
from haystack_integrations.tools.plivo.sms_tool import SendSMSTool
from haystack_integrations.tools.plivo.verify_tool import SendVerificationTool, ValidateVerificationTool


class PlivoToolset(Toolset):
    """
    A :class:`~haystack.tools.Toolset` that bundles all Plivo tools.

    All tools share a single :class:`PlivoClient`, so they authenticate with the same
    account and reuse one ``plivo.RestClient``. The bundled tools are ``send_sms``,
    ``send_verification_code``, ``validate_verification_code``, ``lookup_number``, and
    ``make_call``.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    from haystack_integrations.tools.plivo import PlivoToolset

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=PlivoToolset(sender="+14150000000"),
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
        Create a PlivoToolset.

        :param auth_id: Plivo Auth ID. Defaults to ``Secret.from_env_var("PLIVO_AUTH_ID")``.
        :param auth_token: Plivo Auth Token. Defaults to ``Secret.from_env_var("PLIVO_AUTH_TOKEN")``.
        :param sender: Plivo number or approved sender id in E.164, used for SMS and calls.
        """
        self.client = PlivoClient(auth_id=auth_id, auth_token=auth_token, sender=sender)
        super().__init__(
            tools=[
                SendSMSTool(client=self.client),
                SendVerificationTool(client=self.client),
                ValidateVerificationTool(client=self.client),
                LookupNumberTool(client=self.client),
                MakeCallTool(client=self.client),
            ]
        )

    def warm_up(self) -> None:
        """Build the shared Plivo client (idempotent)."""
        self.client.warm_up()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this toolset to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": self.client.to_dict()["data"],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlivoToolset":
        """Deserialize a PlivoToolset from a dictionary."""
        client = PlivoClient.from_dict({"type": "", "data": data["data"]})
        return cls(auth_id=client.auth_id, auth_token=client.auth_token, sender=client.sender)
