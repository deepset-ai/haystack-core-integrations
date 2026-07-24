# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.tenki.tenki_sandbox import TenkiSandbox


class RunBashCommandTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that executes bash commands inside a Tenki sandbox.

    Pass the same :class:`TenkiSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    from haystack_integrations.tools.tenki import TenkiSandbox, RunBashCommandTool, ReadFileTool

    sandbox = TenkiSandbox()
    agent = Agent(
        chat_generator=...,
        tools=[
            RunBashCommandTool(sandbox=sandbox),
            ReadFileTool(sandbox=sandbox),
        ],
    )
    ```
    """

    def __init__(self, sandbox: TenkiSandbox) -> None:
        """
        Create a RunBashCommandTool.

        :param sandbox: The :class:`TenkiSandbox` instance that will execute commands.
        """

        def run_bash_command(command: str, timeout: int = 60) -> str:
            sb = sandbox._require_sandbox()
            try:
                # check=False: a non-zero exit is a valid result the LLM should
                # see and react to, not an exception.
                result = sb.exec("bash", "-lc", command, timeout=timeout)
            except Exception as e:
                # Only genuine execution/transport failures (e.g. provisioning
                # or data-plane timeouts) land here.
                msg = f"Failed to run bash command: {e}"
                raise RuntimeError(msg) from e

            # Report the full outcome. `result.ok` already accounts for
            # termination by signal, so a SIGKILL (exit_code 0 + signal set)
            # is not mistaken for success. Surface signal/reason so the LLM
            # can distinguish a clean non-zero exit from a killed process.
            lines = [
                f"ok: {result.ok}",
                f"exit_code: {result.exit_code}",
            ]
            if result.signal:
                lines.append(f"signal: {result.signal}")
            if result.reason:
                lines.append(f"reason: {result.reason}")
            lines.append(f"stdout:\n{result.stdout_text}")
            lines.append(f"stderr:\n{result.stderr_text}")
            return "\n".join(lines)

        super().__init__(
            name="run_bash_command",
            description=(
                "Execute a bash command inside the Tenki sandbox and return the combined stdout, "
                "stderr, exit code, and (if the process was killed) the signal. Use this to run "
                "shell scripts, install packages, compile code, or perform any system-level operation."
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
        self._tenki_sandbox = sandbox

    def to_dict(self) -> dict[str, Any]:
        """Serialize this tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"sandbox": self._tenki_sandbox.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunBashCommandTool":
        """Deserialize a RunBashCommandTool from a dictionary."""
        sandbox = TenkiSandbox.from_dict(data["data"]["sandbox"])
        return cls(sandbox=sandbox)
