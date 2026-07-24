# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.tenki.tenki_sandbox import TenkiSandbox


class ReadFileTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that reads files from a Tenki sandbox filesystem.

    Pass the same :class:`TenkiSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.tenki import TenkiSandbox, ReadFileTool

    sandbox = TenkiSandbox()
    agent = Agent(chat_generator=..., tools=[ReadFileTool(sandbox=sandbox)])
    ```
    """

    def __init__(self, sandbox: TenkiSandbox) -> None:
        """
        Create a ReadFileTool.

        :param sandbox: The :class:`TenkiSandbox` instance to read files from.
        """

        def read_file(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                return sb.fs.read_text(path)
            except Exception as e:
                msg = f"Failed to read file '{path}': {e}"
                raise RuntimeError(msg) from e

        super().__init__(
            name="read_file",
            description=(
                "Read the text content of a file from the Tenki sandbox filesystem and return it "
                "as a string. The file must exist; use list_directory to verify paths first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the file to read."}
                },
                "required": ["path"],
            },
            function=read_file,
        )
        self._tenki_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._tenki_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadFileTool":
        """Deserialize a ReadFileTool from a dictionary."""
        sandbox = TenkiSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
