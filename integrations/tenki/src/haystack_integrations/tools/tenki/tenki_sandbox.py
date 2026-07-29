# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading
import uuid
import weakref
from typing import Any, ClassVar

from haystack import logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install tenki'") as tenki_import:
    from tenki import Sandbox

logger = logging.getLogger(__name__)

# Sandbox states that cannot serve a tool call until the VM is resumed. Tenki pauses
# a sandbox after `idle_timeout_minutes` of inactivity.
_PAUSED_STATES = frozenset({"paused", "pausing", "suspended", "idle", "stopped"})


class TenkiSandbox:
    """
    Manages the lifecycle of a Tenki cloud sandbox (microVM).

    Instantiate this class and pass it to one or more Tenki tool classes
    (`RunBashCommandTool`, `ReadFileTool`, `WriteFileTool`,
    `ListDirectoryTool`) to share a single sandbox environment across all
    tools. All tools that receive the same `TenkiSandbox` instance operate
    inside the same live sandbox.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.agents import Agent

    from haystack_integrations.tools.tenki import (
        TenkiSandbox,
        RunBashCommandTool,
        ReadFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )

    sandbox = TenkiSandbox()
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

    Lifecycle is explicit: neither the Agent nor the Pipeline starts or stops the
    sandbox for you. Call :meth:`warm_up` before the first tool invocation and
    :meth:`close` when you are done -- from a ``finally`` block, so a failure
    cannot leave a billed sandbox running:

    ```python
    sandbox.warm_up()
    try:
        ...  # run the agent / use the tools
    finally:
        sandbox.close()
    ```

    :class:`~haystack_integrations.tools.tenki.TenkiToolset` exposes the same
    ``warm_up``/``close`` pair for the sandbox it owns.
    """

    # Process-wide cache used during deserialization to keep tools that
    # shared one sandbox before serialization sharing it after `from_dict`
    # as well. Keyed by `instance_id`. Weak refs so an entry disappears
    # once no tool references the sandbox. A cache hit is only honored
    # when the full serialized config matches (see `from_dict`), so a
    # crafted YAML cannot hijack another tenant's live instance.
    _instances: ClassVar["weakref.WeakValueDictionary[str, TenkiSandbox]"] = weakref.WeakValueDictionary()

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
        instance_id: str | None = None,
    ) -> None:
        """
        Create a TenkiSandbox instance.

        :param auth_token: Tenki auth token or API key. Defaults to reading
            ``TENKI_AUTH_TOKEN`` then ``TENKI_API_KEY`` from the environment.
        :param base_url: Tenki API endpoint. Defaults to the SDK default
            (``https://api.tenki.cloud``) when omitted.
        :param name: Human-readable name for the sandbox session.
        :param cpu_cores: Number of vCPUs to allocate. ``None`` uses the Tenki default.
        :param memory_mb: Memory in MB to allocate. ``None`` uses the Tenki default.
        :param max_duration: Hard upper bound (seconds) on the sandbox's lifetime, after
            which Tenki terminates it. Acts as a backstop so a leaked or forgotten
            sandbox cannot run indefinitely. ``None`` uses the Tenki default.
        :param idle_timeout_minutes: Minutes of inactivity before Tenki pauses the
            sandbox. Raise this for long-running tool sessions so the sandbox is not
            paused mid-task. ``None`` uses the Tenki default.
        :param environment_vars: Optional environment variables to inject into the sandbox.
        :param instance_id: Stable identifier preserved across `to_dict`/`from_dict`. When
            omitted, a fresh UUID is generated. Tools that share the same `TenkiSandbox`
            instance inherit this id, which is what lets them re-share the instance after
            a serialization round-trip. Distinct from the cloud-side sandbox id assigned
            by Tenki at warm-up.
        """
        self.auth_token = auth_token
        self.base_url = base_url
        self.name = name
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.max_duration = max_duration
        self.idle_timeout_minutes = idle_timeout_minutes
        self.environment_vars = environment_vars or {}
        self.instance_id = instance_id or uuid.uuid4().hex
        self._sandbox: Any = None
        # Serializes warm_up so concurrent callers cannot each provision a VM.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Create and start the Tenki sandbox.

        Idempotent and thread-safe -- concurrent callers are serialized, so only one
        microVM is ever created and calling it again while the sandbox is running has
        no effect.

        The creation is failure-atomic and cancellation-safe: if anything goes
        wrong after the microVM is provisioned (including ``CancelledError`` or
        ``KeyboardInterrupt``), the sandbox is torn down so a RUNNING VM is never
        leaked. The VM is created with ``wait=False`` and awaited through
        ``wait_ready()`` inside the guarded block: with the SDK default
        ``wait=True``, a readiness failure raises before a handle is returned and
        the provisioned VM would leak past this cleanup path.

        :raises RuntimeError: If the Tenki sandbox cannot be created.
        """
        if self._sandbox is not None:
            return

        with self._lock:
            # Re-check under the lock: another caller may have created the sandbox
            # while we were waiting for it.
            if self._sandbox is not None:
                return

            tenki_import.check()
            resolved_token = self.auth_token.resolve_value()

            create_kwargs: dict[str, Any] = {
                "name": self.name,
                "auth_token": resolved_token,
                "cpu_cores": self.cpu_cores,
                "memory_mb": self.memory_mb,
                "max_duration": self.max_duration,
                "idle_timeout_minutes": self.idle_timeout_minutes,
                "env": self.environment_vars or None,
                "base_url": self.base_url,
            }
            # Drop unset values so the Tenki SDK applies its own defaults.
            create_kwargs = {k: v for k, v in create_kwargs.items() if v is not None}

            logger.info("Starting Tenki sandbox (name={name})", name=self.name)
            try:
                # Returns as soon as the VM is registered, so readiness is awaited below
                # with a handle already in hand.
                sandbox = Sandbox.create(wait=False, **create_kwargs)
            except Exception as e:
                msg = f"Failed to start Tenki sandbox: {e}"
                raise RuntimeError(msg) from e

            # The VM now exists. Guard everything up to committing the handle: any
            # failure here -- including the readiness wait -- must tear the VM down.
            try:
                sandbox.wait_ready()
                self._sandbox = sandbox
                logger.info("Tenki sandbox started (id={sandbox_id})", sandbox_id=sandbox.id)
            except BaseException as e:
                self._safe_terminate(sandbox)
                self._sandbox = None
                if isinstance(e, Exception):
                    msg = f"Failed to start Tenki sandbox: {e}"
                    raise RuntimeError(msg) from e
                raise

    def close(self) -> None:
        """
        Terminate the Tenki sandbox and release all associated resources.

        Teardown errors are **not** swallowed and the handle is retained on
        failure, so a failed terminate surfaces to the caller and ``close()``
        can be retried. Uses the SDK's state-guarded ``close_if_open`` so
        calling it on an already-terminated sandbox is a safe no-op.
        """
        if self._sandbox is None:
            return
        self._sandbox.close_if_open()
        self._sandbox = None
        logger.info("Tenki sandbox closed")

    @staticmethod
    def _safe_terminate(sandbox: Any) -> None:
        """Best-effort teardown used on the creation-failure/cancellation path."""
        try:
            sandbox.close_if_open()
        except Exception as e:
            # Cleanup path: log but don't mask the original error that triggered teardown.
            logger.warning("Failed to tear down Tenki sandbox during cleanup: {error}", error=e)

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
                "auth_token": self.auth_token.to_dict(),
                "base_url": self.base_url,
                "name": self.name,
                "cpu_cores": self.cpu_cores,
                "memory_mb": self.memory_mb,
                "max_duration": self.max_duration,
                "idle_timeout_minutes": self.idle_timeout_minutes,
                "environment_vars": self.environment_vars,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TenkiSandbox":
        """
        Deserialize a :class:`TenkiSandbox` from a dictionary.

        Multiple tools that shared a single :class:`TenkiSandbox` before serialization
        will share the same restored instance: each tool's `from_dict` consults a
        process-wide cache keyed on `instance_id`. A cache hit is only honored when
        the full serialized config matches the cached entry — a crafted YAML with a
        guessed id but a different config falls through to a fresh instance and never
        observes the cached one.

        :param data: Dictionary created by :meth:`to_dict`.
        :returns: A :class:`TenkiSandbox` instance ready to be warmed up. May be a
            previously-restored instance if the id and config match.
        """
        inner = data["data"]
        instance_id = inner.get("instance_id")

        # Snapshot the incoming config in its serialized (Secret-as-dict) form
        # before `deserialize_secrets_inplace` mutates `inner`, so we can compare
        # against `cached.auth_token.to_dict()` symmetrically.
        incoming_config = {
            "auth_token": inner.get("auth_token"),
            "base_url": inner.get("base_url"),
            "name": inner.get("name", "haystack"),
            "cpu_cores": inner.get("cpu_cores"),
            "memory_mb": inner.get("memory_mb"),
            "max_duration": inner.get("max_duration"),
            "idle_timeout_minutes": inner.get("idle_timeout_minutes"),
            "environment_vars": inner.get("environment_vars", {}),
        }

        if instance_id is not None:
            cached = cls._instances.get(instance_id)
            if cached is not None:
                cached_config = {
                    "auth_token": cached.auth_token.to_dict(),
                    "base_url": cached.base_url,
                    "name": cached.name,
                    "cpu_cores": cached.cpu_cores,
                    "memory_mb": cached.memory_mb,
                    "max_duration": cached.max_duration,
                    "idle_timeout_minutes": cached.idle_timeout_minutes,
                    "environment_vars": cached.environment_vars,
                }
                if incoming_config == cached_config:
                    return cached
                # Id collision with mismatched config: fall through to building
                # a fresh instance, but DO NOT register it — preserves the
                # legitimate cached entry from being evicted.

        deserialize_secrets_inplace(inner, keys=["auth_token"])
        instance = cls(
            auth_token=inner["auth_token"],
            base_url=inner.get("base_url"),
            name=inner.get("name", "haystack"),
            cpu_cores=inner.get("cpu_cores"),
            memory_mb=inner.get("memory_mb"),
            max_duration=inner.get("max_duration"),
            idle_timeout_minutes=inner.get("idle_timeout_minutes"),
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
        """
        Return a usable sandbox handle, resuming it first when Tenki has paused it.

        A non-``None`` handle is not sufficient on its own: after
        ``idle_timeout_minutes`` of inactivity Tenki pauses the microVM, so a handle
        that served the previous tool call can point at a paused sandbox. The
        server-side state is refreshed and the VM resumed before it is handed out.

        :raises RuntimeError: If the sandbox cannot be created or resumed.
        """
        if self._sandbox is None:
            self.warm_up()

        sandbox = self._sandbox
        if sandbox is None:  # pragma: no cover - warm_up raises instead of returning
            msg = "Tenki sandbox is not running. Call warm_up() first."
            raise RuntimeError(msg)

        return self._ensure_running(sandbox)

    @staticmethod
    def _ensure_running(sandbox: Any) -> "Sandbox":
        """
        Refresh the sandbox state and resume the VM when Tenki has paused it.

        Capabilities are looked up defensively so SDK versions that do not expose
        ``refresh``/``resume`` keep the previous behaviour instead of failing here.
        """
        refresh = getattr(sandbox, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as e:
                msg = f"Failed to refresh Tenki sandbox state: {e}"
                raise RuntimeError(msg) from e

        state = str(getattr(sandbox, "state", "") or "").lower()
        if state not in _PAUSED_STATES:
            return sandbox

        resume = getattr(sandbox, "resume", None)
        if not callable(resume):
            msg = f"Tenki sandbox is {state} and this SDK version cannot resume it."
            raise RuntimeError(msg)

        sandbox_id = getattr(sandbox, "id", None)
        logger.info("Resuming paused Tenki sandbox (id={sandbox_id})", sandbox_id=sandbox_id)
        try:
            resume()
            wait_ready = getattr(sandbox, "wait_ready", None)
            if callable(wait_ready):
                wait_ready()
        except Exception as e:
            msg = f"Failed to resume paused Tenki sandbox: {e}"
            raise RuntimeError(msg) from e

        return sandbox
