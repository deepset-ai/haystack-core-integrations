# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from haystack import logging
from haystack.core.serialization import import_class_by_name
from haystack.utils import Secret
from haystack.utils.auth import SecretType

from haystack_integrations.tools.mirage._async import AsyncExecutor
from haystack_integrations.tools.mirage.errors import MirageConfigError
from mirage import MountMode
from mirage import Workspace as _MirageWorkspace
from mirage.resource.registry import REGISTRY, build_resource

logger = logging.getLogger(__name__)

_SECRET_TYPES = {e.value for e in SecretType}

# Marks a serialized OAuth token source inside a mount config (see `_is_token_source`).
_TOKEN_SOURCE_MARKER = "__mirage_token_source__"


def _is_serialized_secret(value: Any) -> bool:
    """Return True if `value` looks like a serialized Haystack `Secret`."""
    return isinstance(value, dict) and value.get("type") in _SECRET_TYPES


def _is_token_source(value: Any) -> bool:
    """
    Duck-typed check for an OAuth *token source* (e.g. deepset's ``OAuthRefreshTokenSource``).

    A token source resolves a bearer access token on demand. Backends whose config accepts a token
    provider callable (such as Mirage's OneDrive `access_token`) can be fed one directly, so the
    integration turns it into a callable at build time and serializes it by reference. Detected
    structurally to avoid a hard dependency on the `oauth-haystack` package.
    """
    return (
        not isinstance(value, Secret)
        and callable(getattr(value, "resolve", None))
        and callable(getattr(value, "to_dict", None))
        and hasattr(value, "requires_subject_token")
    )


def _serialize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Serialize a resource config, converting `Secret`s and OAuth token sources to their dict form."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        if _is_token_source(v):
            out[k] = {_TOKEN_SOURCE_MARKER: v.to_dict()}
        elif isinstance(v, Secret):
            out[k] = v.to_dict()
        else:
            out[k] = v
    return out


def _deserialize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a resource config, restoring `Secret`s and OAuth token sources."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict) and _TOKEN_SOURCE_MARKER in v:
            src_dict = v[_TOKEN_SOURCE_MARKER]
            source_cls: Any = import_class_by_name(src_dict["type"])
            out[k] = source_cls.from_dict(src_dict)
        elif _is_serialized_secret(v):
            out[k] = Secret.from_dict(v)
        else:
            out[k] = v
    return out


def _resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve a mount config into plain values passed to the Mirage backend.

    `Secret`s become their string value. An OAuth token source becomes a zero-arg callable
    (its `resolve` method) that the backend calls on demand to get a fresh bearer token — the
    source handles caching/refresh. Per-request (subject-token) sources are rejected, since the
    integration has no way to supply the per-request subject token.
    """
    out: dict[str, Any] = {}
    for k, v in config.items():
        if _is_token_source(v):
            if getattr(v, "requires_subject_token", False):
                msg = (
                    f"Config key '{k}' is a per-request (subject-token) OAuth source, which the Mirage "
                    "integration cannot supply. Use a config-only source (e.g. OAuthRefreshTokenSource "
                    "or OAuthStaticTokenSource), or resolve a token yourself and pass it as a Secret."
                )
                raise MirageConfigError(msg)
            out[k] = v.resolve  # bound method -> Callable[[], str]; the source caches/refreshes
        elif isinstance(v, Secret):
            out[k] = v.resolve_value()
        else:
            out[k] = v
    return out


@dataclass
class MirageMount:
    """
    Declarative description of a single backend mounted into a :class:`MirageWorkspace`.

    A mount is the serializable unit of a Mirage workspace: it names *where* a backend is mounted
    (`path`), *which* backend it is (`resource`, a Mirage registry name such as `"s3"` or `"gdrive"`),
    and *how* to configure it (`config`).

    `config` values may be plain values, Haystack `Secret` objects for credentials, or an OAuth token
    source (e.g. `OAuthRefreshTokenSource`) for backends whose config accepts a token-provider callable
    (such as Mirage's OneDrive `access_token`). Secrets and token sources are resolved only when the
    live workspace is built.

    Every backend is created the same way. Use the Mirage registry name and the config keys that backend expects
    (discover names with `MirageMount.available_resources()`; config keys come from the backend's Mirage config class):

    ```python
    from haystack.utils import Secret

    MirageMount(path="/data", resource="ram")                                  # in-memory scratch
    MirageMount(path="/local", resource="disk", config={"root": "/srv/data"})  # local disk
    MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True)
    MirageMount(
        path="/drive",
        resource="gdrive",
        config={"client_id": "...", "refresh_token": Secret.from_env_var("GDRIVE_REFRESH_TOKEN")},
        read_only=True,
    )
    ```

    :param path: Mount point in the virtual filesystem, e.g. `"/s3"`.
    :param resource: Mirage registry name of the backend, e.g. `"ram"`, `"disk"`, `"s3"`, `"gdrive"`.
        See `mirage.resource.registry.REGISTRY` or `MirageMount.available_resources()` for the full list.
    :param config: Keyword arguments passed to the backend's Mirage config. Values may be `Secret`s, or
        an OAuth token source that is turned into a token-provider callable when the workspace is built.
    :param read_only: If True, the mount is mounted in Mirage's READ mode and writes are rejected by
        Mirage itself.
    """

    path: str
    resource: str
    config: dict[str, Any] = field(default_factory=dict)
    read_only: bool = False

    @staticmethod
    def available_resources() -> list[str]:
        """
        Return the Mirage registry names usable as `resource`.

        These are short backend names such as `"s3"`, `"gdrive"`, `"postgres"`. Pass one to
        `MirageMount(resource=...)`; the config keys each backend expects come from its Mirage
        config class.
        """
        return sorted(REGISTRY)


class MirageWorkspace:
    """
    A description of a Mirage mount tree that lazily builds a live `mirage.Workspace`.

    `MirageWorkspace` is the shared backend behind the Mirage tools and components: it holds the list of
    :class:`MirageMount`s and the cache configuration, serializes cleanly (resolving `Secret`s only at
    build time), and constructs the live workspace on first use via Mirage's resource registry.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.tools.mirage import MirageWorkspace, MirageMount

    ws = MirageWorkspace(
        mounts=[
            MirageMount(path="/data", resource="ram"),
            MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True),
        ]
    )
    print(ws.run("ls /s3"))
    ```
    """

    def __init__(self, mounts: list[MirageMount], *, cache_limit: str | int = "512MB") -> None:
        """
        Initialize the workspace description.

        :param mounts: The backends to mount, as a list of :class:`MirageMount`.
        :param cache_limit: Mirage file-cache size limit (e.g. `"512MB"` or an int byte count).
        :raises MirageConfigError: If no mounts are provided or mount paths are not unique.
        """
        if not mounts:
            msg = "MirageWorkspace requires at least one mount."
            raise MirageConfigError(msg)
        paths = [m.path for m in mounts]
        if len(set(paths)) != len(paths):
            msg = f"Mount paths must be unique, got: {paths}"
            raise MirageConfigError(msg)
        self.mounts = mounts
        self.cache_limit = cache_limit
        self._live: _MirageWorkspace | None = None
        # Serializes the lazy build so concurrent callers don't each build the live workspace
        self._build_lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------------------------

    def warm_up(self) -> None:
        """Build the live `mirage.Workspace` eagerly. Idempotent."""
        self._ensure_live()

    def _ensure_live(self) -> _MirageWorkspace:
        """Build (once) and return the live `mirage.Workspace`. Thread-safe."""
        # Double-checked locking: the fast path avoids the lock once built; the slow path builds under
        # the lock so concurrent callers can't each construct their own live workspace.
        if self._live is not None:
            return self._live

        with self._build_lock:
            if self._live is not None:
                return self._live

            resources: dict[str, Any] = {}
            for mount in self.mounts:
                try:
                    resource = build_resource(mount.resource, _resolve_config(mount.config))
                except KeyError as e:
                    msg = f"Unknown Mirage resource '{mount.resource}' for mount '{mount.path}'."
                    raise MirageConfigError(msg) from e
                except Exception as e:
                    msg = f"Failed to build Mirage resource '{mount.resource}' for mount '{mount.path}': {e}"
                    raise MirageConfigError(msg) from e
                mode = MountMode.READ if mount.read_only else MountMode.WRITE
                resources[mount.path] = (resource, mode)

            self._live = _MirageWorkspace(resources, cache_limit=self.cache_limit)
            return self._live

    def close(self) -> None:
        """Close the live workspace and release its resources, if it was built. Thread-safe."""
        # Share the build lock so we can't close a workspace another thread is mid-build
        with self._build_lock:
            if self._live is not None:
                try:
                    close_result = self._live.close()
                    # Mirage's Workspace.close() is a coroutine; run it on the background loop.
                    if asyncio.iscoroutine(close_result):
                        AsyncExecutor.get_instance().run(close_result, timeout=10)
                except Exception as e:
                    logger.debug("Error while closing Mirage workspace: {error}", error=str(e))
                finally:
                    self._live = None

    # --- execution ---------------------------------------------------------------------------

    def run(self, command: str, *, timeout: float = 60.0, max_chars: int | None = None) -> str:
        """
        Run a bash `command` against the mount tree from a synchronous context and return its output.

        :param command: A bash command line, e.g. `"grep -r alert /s3/logs | wc -l"`.
        :param timeout: Maximum seconds to wait for the command.
        :param max_chars: If set, truncate the returned text to this many characters.
        :returns: Combined stdout (plus a trailing error note on non-zero exit) as a string.
        """
        live = self._ensure_live()
        result = AsyncExecutor.get_instance().run(live.execute(command), timeout=timeout)
        return _to_text(result, max_chars=max_chars)

    async def run_async(self, command: str, *, timeout: float = 60.0, max_chars: int | None = None) -> str:
        """Async counterpart of :meth:`run`."""
        live = self._ensure_live()
        result = await asyncio.wait_for(live.execute(command), timeout=timeout)
        return _to_text(result, max_chars=max_chars)

    # --- serialization -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workspace description to a dictionary (Secret-safe)."""
        return {
            "mounts": [
                {
                    "path": m.path,
                    "resource": m.resource,
                    "config": _serialize_config(m.config),
                    "read_only": m.read_only,
                }
                for m in self.mounts
            ],
            "cache_limit": self.cache_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MirageWorkspace:
        """Deserialize a workspace description from a dictionary."""
        mounts = [
            MirageMount(
                path=m["path"],
                resource=m["resource"],
                config=_deserialize_config(m.get("config", {})),
                read_only=m.get("read_only", False),
            )
            for m in data["mounts"]
        ]
        return cls(mounts=mounts, cache_limit=data.get("cache_limit", "512MB"))

    def describe(self) -> str:
        """Return a human/LLM-readable summary of the mount tree (used in tool descriptions)."""
        lines = []
        for m in self.mounts:
            access = "read-only" if m.read_only else "read-write"
            lines.append(f"  {m.path}  ->  {m.resource} ({access})")
        return "\n".join(lines)


def _to_text(result: Any, *, max_chars: int | None = None) -> str:
    """
    Normalize a Mirage `IOResult` (or a plain string) into text suitable for an LLM.

    Mirage returns an `IOResult` with `stdout`/`stderr` (bytes) and an `exit_code`. We decode
    stdout, and on a non-zero exit append the exit code and stderr so the agent can self-correct. The
    function falls back to `str(result)`.
    """
    if isinstance(result, str):
        text = result
    elif hasattr(result, "stdout"):
        text = _decode(getattr(result, "stdout", None))
        exit_code = getattr(result, "exit_code", 0) or 0
        if exit_code != 0:
            stderr = _decode(getattr(result, "stderr", None))
            text = f"{text}\n[exit code {exit_code}]" + (f"\n{stderr}" if stderr else "")
    else:
        text = str(result)

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"\n... [output truncated to {max_chars} characters]"
    return text


def _decode(value: Any) -> str:
    """Decode bytes to str (utf-8, replacement on error); pass through str; empty for None."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
