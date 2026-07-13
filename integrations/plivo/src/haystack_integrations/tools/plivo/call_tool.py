# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.plivo.client import PlivoClient


class MakeCallTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that places an outbound phone call through Plivo.

    The call is placed from the ``sender`` configured on the shared :class:`PlivoClient`.
    When the call is answered, Plivo fetches the caller-provided ``answer_url`` for the
    call-flow XML, so you must host that endpoint yourself. This tool only *initiates* a
    call; it does not stream or process audio -- Haystack has no telephony/audio runtime,
    so real-time voice (audio streaming) is out of scope for this integration.

    ### Usage example

    ```python
    from haystack_integrations.tools.plivo import PlivoClient, MakeCallTool

    client = PlivoClient(sender="+14150000000")
    agent = Agent(chat_generator=..., tools=[MakeCallTool(client=client)])
    ```
    """

    def __init__(self, client: PlivoClient) -> None:
        """
        Create a MakeCallTool.

        :param client: The :class:`PlivoClient` used to place the call.
        """

        def make_call(to: str, answer_url: str) -> str:
            rest = client._require_client()
            from_ = client._require_sender()
            try:
                response = rest.calls.create(from_=from_, to_=to, answer_url=answer_url)
            except Exception as e:
                msg = f"Failed to place call to {to}: {e}"
                raise RuntimeError(msg) from e
            request_uuid = getattr(response, "request_uuid", None)
            status = getattr(response, "message", "call queued")
            return f"{status} (request_uuid: {request_uuid})"

        super().__init__(
            name="make_call",
            description=(
                "Place an outbound phone call via Plivo from the account's configured number. "
                "When answered, Plivo fetches the given answer_url for call-flow XML (which you "
                "must host). Returns the queue status and request id. This only starts the call; "
                "it does not stream audio."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Destination phone number in E.164 format, e.g. '+14151234567'.",
                    },
                    "answer_url": {
                        "type": "string",
                        "description": (
                            "Publicly reachable URL that returns Plivo call-flow XML when the call is answered."
                        ),
                    },
                },
                "required": ["to", "answer_url"],
            },
            function=make_call,
        )
        self._plivo_client = client

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"client": self._plivo_client.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MakeCallTool":
        """Deserialize a MakeCallTool from a dictionary."""
        client = PlivoClient.from_dict(data["data"]["client"])
        return cls(client=client)
