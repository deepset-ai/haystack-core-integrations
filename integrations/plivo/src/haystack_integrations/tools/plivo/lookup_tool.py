# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.plivo.client import PlivoClient


class LookupNumberTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that looks up carrier information for a phone number.

    Uses the Plivo Lookup API (``carrier`` type) to report the number's country, line
    type (``landline`` / ``mobile`` / ``voip``), carrier name, and whether it has been
    ported. Carrier fields are best-effort and may be sparse for unallocated numbers.

    ### Usage example

    ```python
    from haystack_integrations.tools.plivo import PlivoClient, LookupNumberTool

    client = PlivoClient()
    agent = Agent(chat_generator=..., tools=[LookupNumberTool(client=client)])
    ```
    """

    def __init__(self, client: PlivoClient) -> None:
        """
        Create a LookupNumberTool.

        :param client: The :class:`PlivoClient` used to perform the lookup.
        """

        def lookup_number(number: str) -> str:
            rest = client._require_client()
            try:
                response = rest.lookup.get(number)
            except Exception as e:
                msg = f"Failed to look up number {number}: {e}"
                raise RuntimeError(msg) from e
            country = getattr(response, "country", None)
            carrier = getattr(response, "carrier", None)
            return (
                f"number: {getattr(response, 'phone_number', number)}, "
                f"country: {getattr(country, 'name', None)}, "
                f"carrier: {getattr(carrier, 'name', None)}, "
                f"type: {getattr(carrier, 'type', None)}, "
                f"ported: {getattr(carrier, 'ported', None)}"
            )

        super().__init__(
            name="lookup_number",
            description=(
                "Look up carrier metadata for a phone number via Plivo Lookup. Returns the country, "
                "line type (landline, mobile, or voip), carrier name, and porting status. Useful to "
                "check whether a number is mobile before sending an SMS."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "Phone number to look up, in E.164 format, e.g. '+14151234567'.",
                    }
                },
                "required": ["number"],
            },
            function=lookup_number,
        )
        self._plivo_client = client

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"client": self._plivo_client.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LookupNumberTool":
        """Deserialize a LookupNumberTool from a dictionary."""
        client = PlivoClient.from_dict(data["data"]["client"])
        return cls(client=client)
