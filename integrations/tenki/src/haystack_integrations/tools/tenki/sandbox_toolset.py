# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.tools.tenki.bash_tool import RunBashCommandTool
from haystack_integrations.tools.tenki.list_directory_tool import ListDirectoryTool
from haystack_integrations.tools.tenki.read_file_tool import ReadFileTool
from haystack_integrations.tools.tenki.tenki_sandbox import TenkiSandbox
from haystack_integrations.tools.tenki.write_file_tool import WriteFileTool


class TenkiToolset(Toolset):
    """
    A :class:`~haystack.tools.Toolset` that bundles all Tenki sandbox tools.

    All tools in the set share a single :class:`TenkiSandbox` instance so they
    operate inside the same live sandbox. The toolset owns the sandbox
    lifecycle: calling :meth:`warm_up` starts the sandbox, and serialisation
    round-trips preserve the shared-sandbox relationship.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.agents import Agent

    from haystack_integrations.tools.tenki import TenkiToolset

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=TenkiToolset(),
    )
    ```
    """

    def __init__(
        self,
        auth_token: Secret = Secret.from_env_var(["TENKI_AUTH_TOKEN", "TENKI_API_KEY"], strict=True),
        base_url: str | None = None,
        name: str = "haystack",
        cpu_cores: int | None = None,
        memory_mb: int | None = None,
        max_duration: float | None = None,
        idle_timeout_minutes: int | None = None,
        environment_vars: dict[str, str] | None = None,
    ) -> None:
        """
        Create a TenkiToolset.

        :param auth_token: Tenki auth token or API key. Defaults to reading
            ``TENKI_AUTH_TOKEN`` then ``TENKI_API_KEY`` from the environment.
        :param base_url: Tenki API endpoint. Defaults to the SDK default when omitted.
        :param name: Human-readable name for the sandbox session.
        :param cpu_cores: Number of vCPUs to allocate. ``None`` uses the Tenki default.
        :param memory_mb: Memory in MB to allocate. ``None`` uses the Tenki default.
        :param max_duration: Hard upper bound (seconds) on the sandbox's lifetime. ``None``
            uses the Tenki default.
        :param idle_timeout_minutes: Minutes of inactivity before Tenki pauses the sandbox.
            ``None`` uses the Tenki default.
        :param environment_vars: Optional environment variables to inject into the sandbox.
        """
        self.sandbox = TenkiSandbox(
            auth_token=auth_token,
            base_url=base_url,
            name=name,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            max_duration=max_duration,
            idle_timeout_minutes=idle_timeout_minutes,
            environment_vars=environment_vars,
        )
        super().__init__(
            tools=[
                RunBashCommandTool(sandbox=self.sandbox),
                ReadFileTool(sandbox=self.sandbox),
                WriteFileTool(sandbox=self.sandbox),
                ListDirectoryTool(sandbox=self.sandbox),
            ]
        )

    def warm_up(self) -> None:
        """Start the shared Tenki sandbox (idempotent)."""
        self.sandbox.warm_up()

    def close(self) -> None:
        """Terminate the shared Tenki sandbox and release cloud resources."""
        self.sandbox.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this toolset to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": self.sandbox.to_dict()["data"],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TenkiToolset":
        """Deserialize a TenkiToolset from a dictionary."""
        inner = data["data"]
        deserialize_secrets_inplace(inner, keys=["auth_token"])
        return cls(
            auth_token=inner["auth_token"],
            base_url=inner.get("base_url"),
            name=inner.get("name", "haystack"),
            cpu_cores=inner.get("cpu_cores"),
            memory_mb=inner.get("memory_mb"),
            max_duration=inner.get("max_duration"),
            idle_timeout_minutes=inner.get("idle_timeout_minutes"),
            environment_vars=inner.get("environment_vars", {}),
        )
