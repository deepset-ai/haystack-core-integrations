# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox


class RunBashCommandTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that executes bash commands inside an E2B sandbox.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.e2b import E2BSandbox, RunBashCommandTool, ReadFileTool

    sandbox = E2BSandbox()
    agent = Agent(
        chat_generator=...,
        tools=[
            RunBashCommandTool(sandbox=sandbox),
            ReadFileTool(sandbox=sandbox),
        ],
    )
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        Create a RunBashCommandTool.

        :param sandbox: The :class:`E2BSandbox` instance that will execute commands.
        """

        def run_bash_command(command: str, timeout: int = 60) -> str:
            sb = sandbox._require_sandbox()
            try:
                result = sb.commands.run(command, timeout=timeout)
                return f"exit_code: {result.exit_code}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            except Exception as e:
                # e2b raises CommandExitException for non-zero exit codes. That exception
                # carries exit_code/stdout/stderr attributes — treat it as a valid result
                # rather than an error so the LLM can see and react to the exit status.
                if hasattr(e, "exit_code"):
                    stdout = getattr(e, "stdout", "")
                    stderr = getattr(e, "stderr", "")
                    return f"exit_code: {e.exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"  # type: ignore[union-attr]
                msg = f"Failed to run bash command: {e}"
                raise RuntimeError(msg) from e

        super().__init__(
            name="run_bash_command",
            description=(
                "Execute a bash command inside the E2B sandbox and return the combined stdout, "
                "stderr, and exit code. Use this to run shell scripts, install packages, compile "
                "code, or perform any system-level operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum number of seconds to wait for the command to finish. Defaults to 60 seconds."
                        ),
                    },
                },
                "required": ["command"],
            },
            function=run_bash_command,
        )
        self._e2b_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._e2b_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunBashCommandTool":
        """Deserialize a RunBashCommandTool from a dictionary."""
        sandbox = E2BSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
