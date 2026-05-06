# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.tools.e2b.bash_tool import RunBashCommandTool
from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox
from haystack_integrations.tools.e2b.list_directory_tool import ListDirectoryTool
from haystack_integrations.tools.e2b.read_file_tool import ReadFileTool
from haystack_integrations.tools.e2b.write_file_tool import WriteFileTool


class E2BToolset(Toolset):
    """
    A :class:`~haystack.tools.Toolset` that bundles all E2B sandbox tools.

    All tools in the set share a single :class:`E2BSandbox` instance so they
    operate inside the same live sandbox process. The toolset owns the sandbox
    lifecycle: calling :meth:`warm_up` starts the sandbox, and serialisation
    round-trips preserve the shared-sandbox relationship.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.agents import Agent

    from haystack_integrations.tools.e2b import E2BToolset

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=E2BToolset(),
    )
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("E2B_API_KEY", strict=True),
        sandbox_template: str = "base",
        timeout: int = 120,
        environment_vars: dict[str, str] | None = None,
    ) -> None:
        """
        Create an E2BToolset.

        :param api_key: E2B API key. Defaults to ``Secret.from_env_var("E2B_API_KEY")``.
        :param sandbox_template: E2B sandbox template name. Defaults to ``"base"``.
        :param timeout: Sandbox inactivity timeout in seconds. Defaults to ``120``.
        :param environment_vars: Optional environment variables to inject into the sandbox.
        """
        self.sandbox = E2BSandbox(
            api_key=api_key,
            sandbox_template=sandbox_template,
            timeout=timeout,
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
        """Start the shared E2B sandbox (idempotent)."""
        self.sandbox.warm_up()

    def close(self) -> None:
        """Shut down the shared E2B sandbox and release cloud resources."""
        self.sandbox.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this toolset to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": self.sandbox.to_dict()["data"],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "E2BToolset":
        """Deserialize an E2BToolset from a dictionary."""
        inner = data["data"]
        deserialize_secrets_inplace(inner, keys=["api_key"])
        return cls(
            api_key=inner["api_key"],
            sandbox_template=inner.get("sandbox_template", "base"),
            timeout=inner.get("timeout", 120),
            environment_vars=inner.get("environment_vars", {}),
        )
