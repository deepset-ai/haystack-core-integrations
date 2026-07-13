# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.plivo.client import PlivoClient


class SendSMSTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that sends an SMS through Plivo.

    The message is sent from the ``sender`` configured on the shared :class:`PlivoClient`,
    so an agent can only send from your account's own number and cannot spoof an arbitrary
    origin.

    ### Usage example

    ```python
    from haystack_integrations.tools.plivo import PlivoClient, SendSMSTool

    client = PlivoClient(sender="+14150000000")
    agent = Agent(chat_generator=..., tools=[SendSMSTool(client=client)])
    ```
    """

    def __init__(self, client: PlivoClient) -> None:
        """
        Create a SendSMSTool.

        :param client: The :class:`PlivoClient` used to send the message.
        """

        def send_sms(to: str, text: str) -> str:
            rest = client._require_client()
            src = client._require_sender()
            try:
                response = rest.messages.create(src=src, dst=to, text=text)
            except Exception as e:
                msg = f"Failed to send SMS to {to}: {e}"
                raise RuntimeError(msg) from e
            message_uuids = getattr(response, "message_uuid", None) or []
            status = getattr(response, "message", "message(s) queued")
            return f"{status} (message_uuid: {', '.join(message_uuids)})"

        super().__init__(
            name="send_sms",
            description=(
                "Send an SMS text message to a phone number via Plivo. The message is sent from "
                "the account's configured sender number. Returns the queue status and message id. "
                "A successful result means the message was queued for delivery, not yet delivered."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format, e.g. '+14151234567'.",
                    },
                    "text": {"type": "string", "description": "The text body of the message to send."},
                },
                "required": ["to", "text"],
            },
            function=send_sms,
        )
        self._plivo_client = client

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"client": self._plivo_client.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SendSMSTool":
        """Deserialize a SendSMSTool from a dictionary."""
        client = PlivoClient.from_dict(data["data"]["client"])
        return cls(client=client)
