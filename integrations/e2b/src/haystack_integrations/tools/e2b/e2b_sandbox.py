# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import uuid
import weakref
from typing import Any, ClassVar

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install e2b'") as e2b_import:
    from e2b import Sandbox

logger = logging.getLogger(__name__)


class E2BSandbox:
    """
    Manages the lifecycle of an E2B cloud sandbox.

    Instantiate this class and pass it to one or more E2B tool classes
    (`RunBashCommandTool`, `ReadFileTool`, `WriteFileTool`,
    `ListDirectoryTool`) to share a single sandbox environment across all
    tools.  All tools that receive the same `E2BSandbox` instance operate
    inside the same live sandbox process.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.agents import Agent

    from haystack_integrations.tools.e2b import (
        E2BSandbox,
        RunBashCommandTool,
        ReadFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )

    sandbox = E2BSandbox()
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=[
            RunBashCommandTool(sandbox=sandbox),
            ReadFileTool(sandbox=sandbox),
            WriteFileTool(sandbox=sandbox),
            ListDirectoryTool(sandbox=sandbox),
        ],
    )
    ```

    Lifecycle is handled automatically by the Agent's pipeline. If you use the
    tools standalone, call :meth:`warm_up` before the first tool invocation:

    ```python
    sandbox.warm_up()
    # ... use tools ...
    sandbox.close()
    ```
    """

    # Process-wide cache used during deserialization to keep tools that
    # shared one sandbox before serialization sharing it after `from_dict`
    # as well. Keyed by `instance_id`. Weak refs so an entry disappears
    # once no tool references the sandbox. A cache hit is only honored
    # when the full serialized config matches (see `from_dict`), so a
    # crafted YAML cannot hijack another tenant's live instance.
    _instances: ClassVar["weakref.WeakValueDictionary[str, E2BSandbox]"] = weakref.WeakValueDictionary()

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("E2B_API_KEY", strict=True),
        sandbox_template: str = "base",
        timeout: int = 120,
        environment_vars: dict[str, str] | None = None,
        instance_id: str | None = None,
    ) -> None:
        """
        Create an E2BSandbox instance.

        :param api_key: E2B API key.
        :param sandbox_template: E2B sandbox template name.
        :param timeout: Sandbox inactivity timeout in seconds.
        :param environment_vars: Optional environment variables to inject into the sandbox.
        :param instance_id: Stable identifier preserved across `to_dict`/`from_dict`. When
            omitted, a fresh UUID is generated. Tools that share the same `E2BSandbox`
            instance inherit this id, which is what lets them re-share the instance after
            a serialization round-trip. Distinct from the cloud-side sandbox id assigned
            by E2B at warm-up.
        """
        self.api_key = api_key
        self.sandbox_template = sandbox_template
        self.timeout = timeout
        self.environment_vars = environment_vars or {}
        self.instance_id = instance_id or uuid.uuid4().hex
        self._sandbox: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Establish the connection to the E2B sandbox.

        Idempotent -- calling it multiple times has no effect if the sandbox is
        already running.

        :raises RuntimeError: If the E2B sandbox cannot be created.
        """
        if self._sandbox is not None:
            return

        e2b_import.check()
        resolved_key = self.api_key.resolve_value()
        try:
            logger.info(
                "Starting E2B sandbox (template={template}, timeout={timeout}s)",
                template=self.sandbox_template,
                timeout=self.timeout,
            )
            self._sandbox = Sandbox.create(
                api_key=resolved_key,
                template=self.sandbox_template,
                timeout=self.timeout,
                envs=self.environment_vars if self.environment_vars else None,
            )
            logger.info("E2B sandbox started (id={sandbox_id})", sandbox_id=self._sandbox.sandbox_id)
        except Exception as e:
            msg = f"Failed to start E2B sandbox: {e}"
            raise RuntimeError(msg) from e

    def close(self) -> None:
        """
        Shut down the E2B sandbox and release all associated resources.

        Call this when you are done to avoid leaving idle sandboxes running.
        """
        if self._sandbox is None:
            return
        try:
            self._sandbox.kill()
            logger.info("E2B sandbox closed")
        except Exception as e:
            logger.warning("Failed to close E2B sandbox: {error}", error=e)
        finally:
            self._sandbox = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the sandbox configuration to a dictionary.

        :returns: Dictionary containing the serialised configuration.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "instance_id": self.instance_id,
                "api_key": self.api_key.to_dict(),
                "sandbox_template": self.sandbox_template,
                "timeout": self.timeout,
                "environment_vars": self.environment_vars,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "E2BSandbox":
        """
        Deserialize an :class:`E2BSandbox` from a dictionary.

        Multiple tools that shared a single :class:`E2BSandbox` before serialization
        will share the same restored instance: each tool's `from_dict` consults a
        process-wide cache keyed on `instance_id`. A cache hit is only honored when
        the full serialized config (api_key, template, timeout, environment_vars)
        matches the cached entry — a crafted YAML with a guessed id but a different
        config falls through to a fresh instance and never observes the cached one.

        :param data: Dictionary created by :meth:`to_dict`.
        :returns: An :class:`E2BSandbox` instance ready to be warmed up. May be a
            previously-restored instance if the id and config match.
        """
        inner = data["data"]
        instance_id = inner.get("instance_id")

        # Snapshot the incoming config in its serialized (Secret-as-dict) form
        # before `deserialize_secrets_inplace` mutates `inner`, so we can compare
        # against `cached.api_key.to_dict()` symmetrically.
        incoming_config = {
            "api_key": inner.get("api_key"),
            "sandbox_template": inner.get("sandbox_template", "base"),
            "timeout": inner.get("timeout", 120),
            "environment_vars": inner.get("environment_vars", {}),
        }

        if instance_id is not None:
            cached = cls._instances.get(instance_id)
            if cached is not None:
                cached_config = {
                    "api_key": cached.api_key.to_dict(),
                    "sandbox_template": cached.sandbox_template,
                    "timeout": cached.timeout,
                    "environment_vars": cached.environment_vars,
                }
                if incoming_config == cached_config:
                    return cached
                # Id collision with mismatched config: fall through to building
                # a fresh instance, but DO NOT register it — preserves the
                # legitimate cached entry from being evicted.

        deserialize_secrets_inplace(inner, keys=["api_key"])
        instance = cls(
            api_key=inner["api_key"],
            sandbox_template=inner.get("sandbox_template", "base"),
            timeout=inner.get("timeout", 120),
            environment_vars=inner.get("environment_vars", {}),
            instance_id=instance_id,
        )
        if instance_id is not None and instance_id not in cls._instances:
            cls._instances[instance_id] = instance
        return instance

    # ------------------------------------------------------------------
    # Internal helpers (used by the tool classes)
    # ------------------------------------------------------------------

    def _require_sandbox(self) -> "Sandbox":
        """Return the active sandbox or raise a helpful error."""
        if self._sandbox is None:
            self.warm_up()
        return self._sandbox
