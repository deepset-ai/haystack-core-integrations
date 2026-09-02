# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Local storage for captured trace artifacts."""

import json
import os
from pathlib import Path
from threading import RLock

from haystack_integrations.agent_pack.tracing.dataclasses import TraceArtifact, TraceSelection


class LocalTraceStore:
    """
    In-memory trace source with optional JSON persistence in a local directory.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        """
        Create a trace store.

        :param directory: Where artifacts are written as one JSON file per run. Artifacts already present are loaded
            on construction. Kept in memory only when omitted.
        """
        self.directory = Path(directory) if directory is not None else None
        self._artifacts: dict[str, TraceArtifact] = {}
        self._lock = RLock()

        # Load existing artifacts from the directory if specified
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.directory.glob("*.json")):
                artifact = TraceArtifact.from_dict(data=json.loads(path.read_text(encoding="utf-8")))
                self._artifacts[artifact.run_id] = artifact

    def add(self, artifact: TraceArtifact) -> None:
        """
        Store or replace an artifact by run ID.

        :param artifact: The artifact to store.
        """
        # We use a lock to ensure thread safety when adding artifacts and writing to disk
        with self._lock:
            self._artifacts[artifact.run_id] = artifact

            # If no directory is specified, we only keep the artifact in memory
            if self.directory is None:
                return

            # We use a temporary file to avoid leaving a partially written file if the process is interrupted
            target = self.directory / f"{artifact.run_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)

    def get(self, run_id: str) -> TraceArtifact:
        """
        Return one artifact by run ID.

        :param run_id: The run to look up.
        :returns: The stored artifact.
        :raises KeyError: If no artifact with that run ID is stored.
        """
        with self._lock:
            try:
                return self._artifacts[run_id]
            except KeyError as error:
                msg = f"Unknown trace run ID: {run_id}."
                raise KeyError(msg) from error

    def list(self, selection: TraceSelection | None = None) -> list[TraceArtifact]:
        """
        Return matching artifacts, newest first.

        :param selection: Which artifacts to return. Every successful run when omitted.
        :returns: The matching artifacts.
        """
        selection = selection or TraceSelection()
        with self._lock:
            artifacts = sorted(self._artifacts.values(), key=lambda artifact: artifact.started_at, reverse=True)
        if selection.run_ids is not None:
            artifacts = [artifact for artifact in artifacts if artifact.run_id in selection.run_ids]
        if selection.status is not None:
            artifacts = [artifact for artifact in artifacts if artifact.status == selection.status]
        return artifacts[: selection.limit] if selection.limit is not None else artifacts
