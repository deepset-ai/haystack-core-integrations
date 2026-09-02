# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Compact input/output records for replaying successful Agent runs."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from haystack.components.agents import Agent
from haystack.utils import _deserialize_value_with_schema, _serialize_value_with_schema


@dataclass(frozen=True, kw_only=True)
class AgentRunRecord:
    """
    Inputs and outputs of one successful Agent run.

    This is deliberately not a tracing abstraction. Optimization needs examples it can replay and compare, not the
    span hierarchy produced while an example ran.

    :param run_id: Stable identifier for the run.
    :param inputs: Keyword arguments passed to ``Agent.run``.
    :param outputs: Dictionary returned by ``Agent.run``.
    """

    run_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the record."""
        return {
            "run_id": self.run_id,
            "inputs": _serialize_value_with_schema(self.inputs),
            "outputs": _serialize_value_with_schema(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRunRecord":
        """Restore a record created by :meth:`to_dict`."""
        inputs = _deserialize_value_with_schema(data["inputs"])
        outputs = _deserialize_value_with_schema(data["outputs"])
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            msg = "Agent run records must deserialize to input and output dictionaries."
            raise ValueError(msg)
        return cls(run_id=data["run_id"], inputs=inputs, outputs=outputs)

    def fingerprint(self) -> str:
        """Return a content fingerprint used to invalidate stale experiment measurements."""
        serialized = self.to_dict()
        payload = json.dumps(
            {"inputs": serialized["inputs"], "outputs": serialized["outputs"]}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RunSelection:
    """Select persisted runs by ID and/or maximum count."""

    run_ids: frozenset[str] | None = None
    limit: int | None = None


class RunSource(Protocol):
    """Source of successful Agent input/output records."""

    def list(self, selection: RunSelection | None = None) -> list[AgentRunRecord]:
        """Return records matching ``selection``."""
        ...


class LocalRunStore:
    """In-memory run source with optional one-JSON-file-per-run persistence."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self._records: dict[str, AgentRunRecord] = {}
        self._lock = RLock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.directory.glob("*.json")):
                record = AgentRunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self._records[record.run_id] = record

    def add(self, record: AgentRunRecord) -> None:
        """Store or replace a run by ID."""
        with self._lock:
            self._records[record.run_id] = record
            if self.directory is None:
                return
            target = self.directory / f"{record.run_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)

    def list(self, selection: RunSelection | None = None) -> list[AgentRunRecord]:
        """Return matching records in insertion order, newest first."""
        selection = selection or RunSelection()
        with self._lock:
            records = list(reversed(self._records.values()))
        if selection.run_ids is not None:
            records = [record for record in records if record.run_id in selection.run_ids]
        return records[: selection.limit] if selection.limit is not None else records


@dataclass(frozen=True, kw_only=True)
class RecordedAgentRun:
    """The live result and persisted record produced by :class:`AgentRunRecorder`."""

    result: dict[str, Any]
    record: AgentRunRecord


class AgentRunRecorder:
    """Run an Agent and retain exactly the inputs and outputs optimization consumes."""

    def __init__(self, store: LocalRunStore | None = None) -> None:
        self.store = store or LocalRunStore()

    def run(self, agent: Agent, **run_kwargs: Any) -> RecordedAgentRun:
        """Run synchronously and persist the successful input/output pair."""
        result = agent.run(**run_kwargs)
        record = AgentRunRecord(run_id=str(uuid4()), inputs=run_kwargs, outputs=result)
        self.store.add(record)
        return RecordedAgentRun(result=result, record=record)

    async def run_async(self, agent: Agent, **run_kwargs: Any) -> RecordedAgentRun:
        """Run asynchronously and persist the successful input/output pair."""
        result = await agent.run_async(**run_kwargs)
        record = AgentRunRecord(run_id=str(uuid4()), inputs=run_kwargs, outputs=result)
        self.store.add(record)
        return RecordedAgentRun(result=result, record=record)


__all__ = [
    "AgentRunRecord",
    "AgentRunRecorder",
    "LocalRunStore",
    "RecordedAgentRun",
    "RunSelection",
    "RunSource",
]
