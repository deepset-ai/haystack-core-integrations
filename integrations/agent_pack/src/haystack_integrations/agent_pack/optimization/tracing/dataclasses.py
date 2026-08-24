# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Dataclasses describing locally captured Agent runs."""

from dataclasses import dataclass
from typing import Any, Literal

TRACE_SCHEMA_VERSION = "haystack-trace/v1"

AGENT_RUN_OPERATION = "haystack.agent.run"
AGENT_INPUT_TAG = "haystack.agent.input"
AGENT_OUTPUT_TAG = "haystack.agent.output"


@dataclass(frozen=True, kw_only=True)
class TraceCaptureLimits:
    """
    Bounds applied to captured span values so trace artifacts stay a manageable size on disk.

    Content tracing records prompts, documents, and embeddings verbatim. Embedding vectors dominate the resulting
    artifact while being useless for replay, so they are dropped by key. The string and sequence bounds are
    deliberately generous: replay inputs are reconstructed from captured content, and truncating a question or a tool
    argument would corrupt them.

    :param max_string_length: Strings longer than this are truncated and marked.
    :param max_sequence_items: Sequences longer than this keep their first items and record how many were dropped.
    :param dropped_keys: Mapping keys omitted entirely wherever they appear.
    """

    max_string_length: int = 32_768
    max_sequence_items: int = 1_024
    dropped_keys: tuple[str, ...] = ("embedding", "sparse_embedding")


DEFAULT_TRACE_CAPTURE_LIMITS = TraceCaptureLimits()


@dataclass(frozen=True, kw_only=True)
class TraceArtifact:
    """
    A locally captured run in the `haystack-trace/v1` shape.

    :param run_id: Identifier of the captured run.
    :param started_at: ISO 8601 timestamp of when the run started.
    :param finished_at: ISO 8601 timestamp of when the run finished.
    :param duration_ms: Wall-clock duration of the run.
    :param status: Whether the run succeeded or raised.
    :param traces: The captured span records.
    :param failure: Type, message, and stacktrace of the failure, for a failed run.
    :param logs: Reserved for captured log records. Log capture is not implemented, so this is always empty.
    :param schema_version: The artifact schema this record conforms to.
    """

    run_id: str
    started_at: str
    finished_at: str
    duration_ms: float
    status: Literal["success", "failed"]
    traces: tuple[dict[str, Any], ...]
    failure: dict[str, Any] | None = None
    logs: tuple[dict[str, Any], ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the TraceArtifact into a dictionary.

        Span records are already bounded and JSON-safe when they come from `RunCaptureTracer`; they are written
        through unchanged so truncation markers are not applied twice.

        :returns: A dictionary with one key per field, with the span and log tuples as lists.
        """
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "traces": [dict(span) for span in self.traces],
            "logs": [dict(record) for record in self.logs],
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceArtifact":
        """
        Create a new TraceArtifact object from a dictionary.

        :param data: The dictionary to build the artifact from.
        :returns: The created object.
        :raises ValueError: If the dictionary declares an unsupported schema version.
        """
        if data.get("schema_version") != TRACE_SCHEMA_VERSION:
            msg = f"Unsupported trace schema: {data.get('schema_version')!r}."
            raise ValueError(msg)
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            duration_ms=float(data["duration_ms"]),
            status=data["status"],
            traces=tuple(data.get("traces") or ()),
            logs=tuple(data.get("logs") or ()),
            failure=data.get("failure"),
        )


@dataclass(frozen=True, kw_only=True)
class TraceSelection:
    """
    Selection criteria shared by local and future remote trace sources.

    :param run_ids: Restrict the selection to these run IDs.
    :param status: Restrict the selection to runs with this status. Successful runs by default.
    :param limit: Return at most this many artifacts, newest first.
    """

    run_ids: frozenset[str] | None = None
    status: Literal["success", "failed"] | None = "success"
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            msg = "TraceSelection.limit must be at least 1."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class CapturedAgentRun:
    """
    Result and trace artifact produced by `TraceCapturingAgentRunner`.

    :param result: The dictionary the Agent run returned.
    :param trace: The captured trace artifact.
    """

    result: dict[str, Any]
    trace: TraceArtifact
