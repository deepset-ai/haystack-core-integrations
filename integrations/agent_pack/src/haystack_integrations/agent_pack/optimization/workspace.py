# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path
from threading import RLock
from typing import Any

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.dataclasses import CandidateConfiguration
from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    content_digest,
    load_agent,
)


class ConfigurationWorkspace:
    """
    Holds the configuration the optimizer edits during an experiment.

    The configuration lives in one YAML file, which the optimizer reads, edits, validates and submits through the
    workspace tools in `optimization/tools.py`. The reference and every submitted candidate are kept as snapshots,
    and `begin_turn` starts each proposal turn from one of them.
    """

    def __init__(
        self,
        path: str | Path,
        reference_yaml: str,
        validator: Callable[[Agent | Pipeline], None] | None = None,
        loader: Callable[[str], Agent | Pipeline] = load_agent,
    ) -> None:
        """
        Use an existing YAML draft, or initialize a new file from the reference.

        :param path: The only file the editing tools can write.
        :param reference_yaml: The reference configuration, as one Pipeline YAML document.
        :param validator: Optional evaluator-specific check after deserialization.
        :param loader: Builds the configuration from YAML, and decides what shape a candidate must keep. Defaults
            to the Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.
        :raises ValueError: If `path` is a symlink.
        """
        self.path = Path(path)
        if self.path.is_symlink():
            msg = "The configuration file must not be a symlink."
            raise ValueError(msg)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(reference_yaml)
        self.loader = loader
        self.reference_id = _configuration_id(reference_yaml)
        self.snapshots = {self.reference_id: reference_yaml}
        self.parent_id = self.reference_id
        self.validator = validator
        self.validated_revision: str | None = None
        self.submitted: CandidateConfiguration | None = None
        self.finished = False
        self.finish_reason: str | None = None
        self.validation_failures: list[dict[str, str]] = []
        self._lock = RLock()

    def _read(self) -> str:
        """Return the file's text."""
        return self.path.read_text(encoding="utf-8")

    def _write(self, text: str, expected_revision: str) -> dict[str, str]:
        """Overwrite the file if the turn is open and `expected_revision` is current, and return the new revision."""
        if self.submitted is not None or self.finished:
            msg = "This proposal turn has ended."
            raise ValueError(msg)
        if content_digest(payload=self._read()) != expected_revision:
            msg = "Stale revision: read_config again before editing."
            raise ValueError(msg)
        self._overwrite(text=text)
        self.validated_revision = None
        return {"revision": content_digest(payload=text)}

    def _overwrite(self, text: str) -> None:
        """Write a temporary file and rename it over `path`, so a symlink there is replaced, not written through."""
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, self.path)

    def _read_config(self) -> dict[str, str]:
        """Return the current YAML, its revision and the snapshot it was edited from."""
        with self._lock:
            text = self._read()
            return {"yaml": text, "revision": content_digest(payload=text), "parent_id": self.parent_id}

    def _edit_config(self, old: str, new: str, expected_revision: str) -> dict[str, str]:
        """Replace the one occurrence of `old` with `new` and return the new revision."""
        with self._lock:
            text = self._read()
            if not old or text.count(old) != 1:
                msg = "old must match exactly once; include more surrounding text to disambiguate."
                raise ValueError(msg)
            return self._write(text=text.replace(old, new, 1), expected_revision=expected_revision)

    def _validate_config(self) -> dict[str, Any]:
        """Load the current YAML with `loader` and run `validator` on it, recording the revision when it passes."""
        with self._lock:
            text = self._read()
            revision = content_digest(payload=text)
            self.validated_revision = None
            try:
                loaded = self.loader(text)
                try:
                    if self.validator is not None:
                        self.validator(loaded)
                    # A Pipeline that is not an Agent has no tools
                    tools = getattr(loaded, "tools", None) or []
                    specs = [item.tool_spec for item in flatten_tools_or_toolsets(tools)]
                finally:
                    loaded.close()
            except Exception as error:
                failure = {"revision": revision, "error": f"{type(error).__name__}: {error}"}
                self.validation_failures.append(failure)
                return {"valid": False, **failure}
            self.validated_revision = revision
            return {"valid": True, "revision": revision, "tools": specs}

    def _submit_candidate(self, expected_revision: str, rationale: str) -> dict[str, str]:
        """Snapshot the validated revision as `submitted` and end the turn."""
        with self._lock:
            text = self._read()
            if self.finished or self.submitted is not None:
                msg = "This proposal turn has ended."
                raise ValueError(msg)
            if expected_revision != content_digest(payload=text) or self.validated_revision != expected_revision:
                msg = "Validate the current revision before submitting."
                raise ValueError(msg)
            candidate_id = _configuration_id(text)
            if candidate_id in self.snapshots:
                msg = "duplicate_or_no_op: this configuration was already submitted or is the reference."
                raise ValueError(msg)
            diff = "".join(
                unified_diff(
                    self.snapshots[self.parent_id].splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile=self.parent_id,
                    tofile=candidate_id,
                )
            )
            self.submitted = CandidateConfiguration(
                candidate_id=candidate_id, parent_id=self.parent_id, yaml=text, rationale=rationale, diff=diff
            )
            self.snapshots[candidate_id] = text
            return {"candidate_id": candidate_id}

    def _restore_candidate(self, candidate_id: str, expected_revision: str) -> dict[str, str]:
        """Reset the file to a snapshot, `"reference"` included, and make it the parent of further edits."""
        with self._lock:
            key = self.reference_id if candidate_id == "reference" else candidate_id
            if key not in self.snapshots:
                msg = "Unknown candidate ID."
                raise ValueError(msg)
            result = self._write(text=self.snapshots[key], expected_revision=expected_revision)
            self.parent_id = key
            return result

    def _finish(self, reason: str) -> str:
        """End the experiment without a submission, recording `reason`."""
        with self._lock:
            if self.submitted is None:
                self.finished = True
                self.finish_reason = reason
            return "Finished."

    def begin_turn(self, base_id: str | None = None) -> None:
        """
        Start a new proposal turn, keeping every snapshot available to restore.

        :param base_id: The snapshot the next edits start from; the file is reset to it. When it is None or not a
            known snapshot, the turn continues from the last submitted candidate, or from the current file when
            nothing was submitted.
        """
        with self._lock:
            if base_id is not None and base_id in self.snapshots:
                if self._read() != self.snapshots[base_id]:
                    self._overwrite(text=self.snapshots[base_id])
                self.parent_id = base_id
            elif self.submitted is not None:
                self.parent_id = self.submitted.candidate_id
            self.submitted = None
            self.validated_revision = None
