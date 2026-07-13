# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.plivo.client import PlivoClient

VALIDATION_SUCCESS_MESSAGE = "session validated successfully"


class SendVerificationTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that starts a Plivo Verify (OTP) session.

    Plivo sends a one-time code to the recipient over SMS or voice and returns a
    ``session_uuid``. Keep that id and pass it to :class:`ValidateVerificationTool`
    together with the code the user enters.

    ### Usage example

    ```python
    from haystack_integrations.tools.plivo import PlivoClient, SendVerificationTool

    client = PlivoClient()
    agent = Agent(chat_generator=..., tools=[SendVerificationTool(client=client)])
    ```
    """

    def __init__(self, client: PlivoClient) -> None:
        """
        Create a SendVerificationTool.

        :param client: The :class:`PlivoClient` used to create the verification session.
        """

        def send_verification(recipient: str, channel: str = "sms") -> str:
            rest = client._require_client()
            try:
                response = rest.verify_session.create(recipient=recipient, channel=channel)
            except Exception as e:
                msg = f"Failed to start verification for {recipient}: {e}"
                raise RuntimeError(msg) from e
            session_uuid = getattr(response, "session_uuid", None)
            status = getattr(response, "message", "")
            return f"Verification code sent to {recipient} via {channel}. session_uuid: {session_uuid} ({status})"

        super().__init__(
            name="send_verification_code",
            description=(
                "Send a one-time verification code (OTP) to a phone number via Plivo Verify over "
                "SMS or voice. Returns a session_uuid that must be passed to validate_verification_code "
                "to check the code the user enters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format, e.g. '+14151234567'.",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["sms", "voice"],
                        "description": "Delivery channel for the code. Defaults to 'sms'.",
                    },
                },
                "required": ["recipient"],
            },
            function=send_verification,
        )
        self._plivo_client = client

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"client": self._plivo_client.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SendVerificationTool":
        """Deserialize a SendVerificationTool from a dictionary."""
        client = PlivoClient.from_dict(data["data"]["client"])
        return cls(client=client)


class ValidateVerificationTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that validates a Plivo Verify code against its session.

    Success is decided solely on the API response message: the code is treated as valid
    only when Plivo returns ``"session validated successfully"``. Any other outcome -- a
    wrong or expired code, or an SDK error -- is reported as *not verified*, so the tool
    never reports a false positive to the agent.

    ### Usage example

    ```python
    from haystack_integrations.tools.plivo import PlivoClient, ValidateVerificationTool

    client = PlivoClient()
    agent = Agent(chat_generator=..., tools=[ValidateVerificationTool(client=client)])
    ```
    """

    def __init__(self, client: PlivoClient) -> None:
        """
        Create a ValidateVerificationTool.

        :param client: The :class:`PlivoClient` used to validate the code.
        """

        def validate_verification(session_uuid: str, otp: str) -> str:
            rest = client._require_client()
            try:
                response = rest.verify_session.validate(session_uuid, otp=otp)
            except Exception as e:
                # A wrong or expired code surfaces as an SDK error; fail closed rather
                # than raising, and never report it as verified.
                return f"not verified (session {session_uuid}): {e}"
            message = getattr(response, "message", "")
            verified = message == VALIDATION_SUCCESS_MESSAGE
            return f"{'verified' if verified else 'not verified'} (session {session_uuid}): {message}"

        super().__init__(
            name="validate_verification_code",
            description=(
                "Validate a one-time verification code (OTP) that a user entered, against the "
                "session_uuid returned by send_verification_code. Returns whether the code is "
                "'verified' or 'not verified'. Only a genuine match is reported as verified."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_uuid": {
                        "type": "string",
                        "description": "The session_uuid returned by send_verification_code.",
                    },
                    "otp": {"type": "string", "description": "The one-time code the user entered."},
                },
                "required": ["session_uuid", "otp"],
            },
            function=validate_verification,
        )
        self._plivo_client = client

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"client": self._plivo_client.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidateVerificationTool":
        """Deserialize a ValidateVerificationTool from a dictionary."""
        client = PlivoClient.from_dict(data["data"]["client"])
        return cls(client=client)
