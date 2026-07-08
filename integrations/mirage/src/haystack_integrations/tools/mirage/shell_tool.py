# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool

from haystack_integrations.tools.mirage.errors import MirageCommandNotAllowedError
from haystack_integrations.tools.mirage.workspace import MirageWorkspace
from mirage.shell.parse import parse as _parse_bash

logger = logging.getLogger(__name__)


def _command_names_in(command: str) -> list[str]:
    """
    Return the literal name of every command Mirage would run in `command`.

    Walks the full tree-sitter AST, so command names nested inside command substitutions `$(...)`,
    backticks, process substitutions `<(...)`, subshells, pipelines, and control flow are all
    surfaced - not only the leading command of each top-level segment. Names that cannot be resolved
    statically (built from a variable or another substitution, e.g. `$X`, or a quoted `"rm"`) are
    returned verbatim; since they will not match a plain allowlist entry, the guard rejects them,
    i.e. it fails closed.
    """
    root = _parse_bash(command)
    names: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "command":
            for child in node.named_children:
                if child.type == "command_name":
                    text = child.text.decode(errors="replace") if child.text else ""
                    if text:
                        names.append(text)
                    break
        stack.extend(node.children)
    return names


def _describe_tool(workspace: MirageWorkspace) -> str:
    """Build a default tool description that tells the LLM what it can do and which mounts exist."""
    return (
        "Run a bash command line against a virtual filesystem that unifies multiple backends. "
        "You can use familiar Unix tools (ls, cat, grep, head, wc, cp, ...) and pipes across mounts. "
        "Available mounts:\n"
        f"{workspace.describe()}"
    )


class MirageShellTool(Tool):
    """
    A Haystack `Tool` that lets an `Agent` run bash commands across a Mirage virtual filesystem.

    Mirage mounts heterogeneous backends (object storage, databases, SaaS apps, local disk) as one
    filesystem; this tool exposes Mirage's single `execute` surface to an Agent as one well-described
    tool with a `command` parameter. Output is normalized to text and truncated before it reaches the
    model.

    ### Security model

    Mirage never shells out to the host: every command runs inside Mirage's own virtual-filesystem
    interpreter, so the blast radius is confined to the mounts you attach. Two controls shape what an
    Agent can do:

    - **Per-mount read-only mode** (``MirageMount(..., read_only=True)``) is the authoritative write
      boundary. Mirage refuses any write to a read-only mount regardless of the command used, so this
      -- not the allowlist -- is how you prevent modification or deletion. Mount anything the Agent
      should not change as read-only.
    - **The command allowlist** (``allowed_commands``) restricts *which* commands may run. It is
      enforced against every command Mirage would execute, including commands nested inside
      ``$(...)``, backticks, ``<(...)`` and subshells, so ``ls "$(rm x)"`` is rejected unless ``rm``
      is also allowed. Treat it as a best-effort filter to steer the Agent, not a sandbox: allowing a
      command that itself runs other commands (``eval``, ``bash``, ``sh``, ``source``, ``xargs``,
      ``timeout``) effectively allows anything, so do not list those for untrusted/hosted use.
    - **``denied_paths``** rejects any command whose text references one of the given path substrings.

    ### Usage example
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.mirage import MirageWorkspace, MirageMount, MirageShellTool

    workspace = MirageWorkspace([
        MirageMount(path="/data", resource="ram"),
        MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True),
    ])
    tool = MirageShellTool(workspace, allowed_commands=["ls", "cat", "grep", "head", "wc", "cp"])

    agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[tool])
    result = agent.run(messages=[ChatMessage.from_user("How many lines in /s3/log.txt mention 'alert'?")])
    print(result["messages"][-1].text)
    ```
    """

    def __init__(
        self,
        workspace: MirageWorkspace,
        *,
        name: str = "mirage_shell",
        description: str | None = None,
        invocation_timeout: float = 60.0,
        max_output_chars: int = 20_000,
        allowed_commands: list[str] | None = None,
        denied_paths: list[str] | None = None,
    ) -> None:
        """
        Initialize the Mirage shell tool.

        :param workspace: The :class:`MirageWorkspace` describing the mount tree.
        :param name: Tool name exposed to the LLM.
        :param description: Custom description. If None, one is generated from the mount tree.
        :param invocation_timeout: Maximum seconds to wait for a command to finish.
        :param max_output_chars: Truncate command output to this many characters before returning it.
        :param allowed_commands: If set, only these command names may run, e.g.
            ``["ls", "cat", "grep", "head", "wc"]``. The allowlist is enforced against *every* command
            Mirage would execute -- including commands nested in substitutions/subshells -- so
            ``ls "$(rm x)"`` is rejected unless ``rm`` is also allowed. It is a filter over Mirage's
            virtual commands to steer the Agent, not a security sandbox; the write boundary is
            per-mount ``read_only`` (see the class "Security model" section). If None, any command is
            allowed (not recommended for untrusted/hosted use).
        :param denied_paths: If set, any command referencing one of these path substrings is rejected.
        """
        self._workspace = workspace
        self._invocation_timeout = invocation_timeout
        self._max_output_chars = max_output_chars
        self._allowed_commands = allowed_commands
        self._denied_paths = denied_paths

        parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command line to run across the mounted filesystem.",
                }
            },
            "required": ["command"],
        }
        super().__init__(
            name=name,
            description=description or _describe_tool(workspace),
            parameters=parameters,
            function=self._invoke,
        )

    def warm_up(self) -> None:
        """Build the underlying live workspace eagerly. Called by `Agent.warm_up()`/`Pipeline.warm_up()`."""
        self._workspace.warm_up()

    def _guard(self, command: str) -> None:
        """
        Enforce the denied-path rules and command allowlist before executing `command`.

        The allowlist is checked against every command Mirage would run, including commands nested in
        substitutions and subshells (see :func:`_command_names_in`).
        """
        if self._denied_paths:
            for denied in self._denied_paths:
                if denied in command:
                    msg = f"Command references a denied path '{denied}'."
                    raise MirageCommandNotAllowedError(msg)

        if self._allowed_commands is None:
            return

        allowed = set(self._allowed_commands)
        names = _command_names_in(command)
        # If an allowlist is set but no command name can be resolved (e.g. a bare redirect like `> /data/x`),
        # there is nothing to match against the allowlist, so reject rather than let it through.
        if not names:
            self._reject_command("<no resolvable command>", allowed)
        for base in names:
            if base not in allowed:
                self._reject_command(base, allowed)

        return

    @staticmethod
    def _reject_command(base: str, allowed: set[str]) -> None:
        """Raise a uniform "command not allowed" error for `base`."""
        msg = f"Command '{base}' is not allowed. Allowed commands: {sorted(allowed)}. (rejected before execution)"
        raise MirageCommandNotAllowedError(msg)

    def _invoke(self, command: str) -> str:
        """Guard, then run the command on the workspace and return normalized, truncated text."""
        self._guard(command)
        return self._workspace.run(command, timeout=self._invocation_timeout, max_chars=self._max_output_chars)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary in the `{"type": ..., "data": ...}` format."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "workspace": self._workspace.to_dict(),
                "name": self.name,
                "description": self.description,
                "invocation_timeout": self._invocation_timeout,
                "max_output_chars": self._max_output_chars,
                "allowed_commands": self._allowed_commands,
                "denied_paths": self._denied_paths,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirageShellTool":
        """Deserialize the tool from a dictionary."""
        inner = data["data"]
        return cls(
            workspace=MirageWorkspace.from_dict(inner["workspace"]),
            name=inner.get("name", "mirage_shell"),
            description=inner.get("description"),
            invocation_timeout=inner.get("invocation_timeout", 60.0),
            max_output_chars=inner.get("max_output_chars", 20_000),
            allowed_commands=inner.get("allowed_commands"),
            denied_paths=inner.get("denied_paths"),
        )

    def close(self) -> None:
        """Close the underlying workspace."""
        if hasattr(self, "_workspace") and self._workspace is not None:
            self._workspace.close()

    def __del__(self) -> None:
        self.close()
