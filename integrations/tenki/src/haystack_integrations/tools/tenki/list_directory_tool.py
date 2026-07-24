# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import posixpath
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.tenki.tenki_sandbox import TenkiSandbox


class ListDirectoryTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that lists directory contents in a Tenki sandbox.

    Pass the same :class:`TenkiSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.tenki import TenkiSandbox, ListDirectoryTool

    sandbox = TenkiSandbox()
    agent = Agent(chat_generator=..., tools=[ListDirectoryTool(sandbox=sandbox)])
    ```
    """

    def __init__(self, sandbox: TenkiSandbox) -> None:
        """
        Create a ListDirectoryTool.

        :param sandbox: The :class:`TenkiSandbox` instance to list directories from.
        """

        def list_directory(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                entries = sb.fs.list(path)
            except Exception as e:
                msg = f"Failed to list directory '{path}': {e}"
                raise RuntimeError(msg) from e

            lines = []
            for entry in entries:
                # FileInfo exposes a full `path`; show just the entry name and
                # mark directories with a trailing slash.
                name = posixpath.basename(entry.path.rstrip("/")) or entry.path
                if entry.is_dir:
                    name += "/"
                lines.append(name)
            return "\n".join(lines) if lines else "(empty directory)"

        super().__init__(
            name="list_directory",
            description=(
                "List the files and subdirectories inside a directory in the Tenki sandbox "
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
        self._tenki_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._tenki_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListDirectoryTool":
        """Deserialize a ListDirectoryTool from a dictionary."""
        sandbox = TenkiSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
