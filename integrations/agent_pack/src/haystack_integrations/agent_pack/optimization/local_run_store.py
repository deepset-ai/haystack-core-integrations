# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from threading import RLock

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord


class LocalRunStore:
    """In-memory run store with optional one-JSON-file-per-run persistence."""

    def __init__(self, directory: str | Path | None = None) -> None:
        """
        Load an optional directory of persisted run records.

        :param directory: Directory containing one JSON file per run. When omitted, records exist only in memory.
            The directory is created when it does not exist.
        """
        self.directory = Path(directory) if directory is not None else None
        self._records: dict[str, AgentRunRecord] = {}
        self._lock = RLock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.directory.glob("*.json")):
                record = AgentRunRecord.from_dict(data=json.loads(path.read_text(encoding="utf-8")))
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

    def clear(self) -> None:
        """
        Remove every stored run, from memory and from disk.

        A recorded run belongs to the Agent that produced it, and nothing about a record says which Agent that was.
        Clearing before recording is what keeps a run store describing the Agent currently under test rather than
        one that has since been reconfigured.
        """
        with self._lock:
            self._records.clear()
            if self.directory is None:
                return
            for path in self.directory.glob("*.json"):
                path.unlink()

    def list(self, run_ids: frozenset[str] | None = None) -> list[AgentRunRecord]:
        """
        Return matching records in insertion order, newest first.

        :param run_ids: Optional run identifiers to include. When omitted, all records are returned.
        :returns: Matching run records, newest first.
        """
        with self._lock:
            records = list(reversed(self._records.values()))
        return records if run_ids is None else [record for record in records if record.run_id in run_ids]
