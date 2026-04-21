# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox


class WriteFileTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that writes files to an E2B sandbox filesystem.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.e2b import E2BSandbox, WriteFileTool

    sandbox = E2BSandbox()
    agent = Agent(chat_generator=..., tools=[WriteFileTool(sandbox=sandbox)])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        Create a WriteFileTool.

        :param sandbox: The :class:`E2BSandbox` instance to write files to.
        """

        def write_file(path: str, content: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                sb.files.write(path, content)
                return f"File written successfully: {path}"
            except Exception as e:
                msg = f"Failed to write file '{path}': {e}"
                raise RuntimeError(msg) from e

        super().__init__(
            name="write_file",
            description=(
                "Write text content to a file in the E2B sandbox filesystem. "
                "Parent directories are created automatically if they do not exist. "
                "Existing files are overwritten."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the file to write."},
                    "content": {"type": "string", "description": "Text content to write into the file."},
                },
                "required": ["path", "content"],
            },
            function=write_file,
        )
        self._e2b_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._e2b_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteFileTool":
        """Deserialize a WriteFileTool from a dictionary."""
        sandbox = E2BSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
