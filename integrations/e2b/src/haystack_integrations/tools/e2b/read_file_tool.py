# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox


class ReadFileTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that reads files from an E2B sandbox filesystem.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.e2b import E2BSandbox, ReadFileTool

    sandbox = E2BSandbox()
    agent = Agent(chat_generator=..., tools=[ReadFileTool(sandbox=sandbox)])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        Create a ReadFileTool.

        :param sandbox: The :class:`E2BSandbox` instance to read files from.
        """

        def read_file(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                content = sb.files.read(path)
                if isinstance(content, bytes):
                    return content.decode("utf-8", errors="replace")
                return str(content)
            except Exception as e:
                msg = f"Failed to read file '{path}': {e}"
                raise RuntimeError(msg) from e

        super().__init__(
            name="read_file",
            description=(
                "Read the text content of a file from the E2B sandbox filesystem and return it "
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
        self._e2b_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._e2b_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadFileTool":
        """Deserialize a ReadFileTool from a dictionary."""
        sandbox = E2BSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
