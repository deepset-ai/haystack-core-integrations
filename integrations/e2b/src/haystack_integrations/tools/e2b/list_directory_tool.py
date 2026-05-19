# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox


class ListDirectoryTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that lists directory contents in an E2B sandbox.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.e2b import E2BSandbox, ListDirectoryTool

    sandbox = E2BSandbox()
    agent = Agent(chat_generator=..., tools=[ListDirectoryTool(sandbox=sandbox)])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        Create a ListDirectoryTool.

        :param sandbox: The :class:`E2BSandbox` instance to list directories from.
        """

        def list_directory(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                entries = sb.files.list(path)
                lines = []
                for entry in entries:
                    name = entry.name
                    if getattr(entry, "is_dir", False) or getattr(entry, "type", "") == "dir":
                        name = name + "/"
                    lines.append(name)
                return "\n".join(lines) if lines else "(empty directory)"
            except Exception as e:
                msg = f"Failed to list directory '{path}': {e}"
                raise RuntimeError(msg) from e

        super().__init__(
            name="list_directory",
            description=(
                "List the files and subdirectories inside a directory in the E2B sandbox "
                "filesystem. Returns a newline-separated list of names with a trailing '/' "
                "appended to subdirectory names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the directory to list."}
                },
                "required": ["path"],
            },
            function=list_directory,
        )
        self._e2b_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._e2b_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListDirectoryTool":
        """Deserialize a ListDirectoryTool from a dictionary."""
        sandbox = E2BSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
