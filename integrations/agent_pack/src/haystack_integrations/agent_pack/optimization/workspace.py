# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.tools import Tool, flatten_tools_or_toolsets
from haystack.tools.from_function import create_tool_from_function

from haystack_integrations.agent_pack.optimization.dataclasses import CandidateConfiguration
from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    content_digest,
    load_agent,
)


class ConfigurationWorkspace:
    """
    Holds the configuration the optimizer edits during an experiment.

    The configuration lives in one YAML file. `tools()` returns the tools the optimizer uses to read, edit, validate
    and submit it. The reference and every submitted candidate are kept as snapshots, and `begin_turn` starts each
    proposal turn from one of them.
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
        """Read the entire editable YAML and its revision for subsequent edits."""
        with self._lock:
            text = self._read()
            return {"yaml": text, "revision": content_digest(payload=text), "parent_id": self.parent_id}

    def _edit_config(
        self,
        old: Annotated[
            str,
            "Nonempty text to replace, matched literally and occurring exactly once in the current YAML. Include "
            "enough surrounding lines to be unique: a bare 'top_k: 2' or a type line repeated across components "
            "matches more than once and is rejected. Pass the entire YAML to rewrite the whole file.",
        ],
        new: Annotated[str, "Text replacing that block verbatim, or empty text to delete it."],
        expected_revision: Annotated[
            str, "The revision returned by read_config or by the preceding edit, which must still be current."
        ],
    ) -> dict[str, str]:
        """Replace one exact text block. Use the entire current YAML as old for a full rewrite."""
        with self._lock:
            text = self._read()
            if not old or text.count(old) != 1:
                msg = "old must match exactly once; include more surrounding text to disambiguate."
                raise ValueError(msg)
            return self._write(text=text.replace(old, new, 1), expected_revision=expected_revision)

    def _validate_config(self) -> dict[str, Any]:
        """Check the YAML loads as the expected Agent or Pipeline and passes the evaluator's checks, without running."""
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

    def _submit_candidate(
        self,
        expected_revision: Annotated[
            str, "The revision returned by a successful validate_config, which must still be current."
        ],
        rationale: Annotated[
            str,
            "The hypothesis this candidate tests: what was changed and what it is expected to move. Read back "
            "alongside the score, so name the change rather than restating the goal.",
        ],
    ) -> dict[str, str]:
        """Submit this validated revision for evaluation and end the proposal turn."""
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

    def _restore_candidate(
        self,
        candidate_id: Annotated[
            str, "A candidate ID from the outcomes so far, or 'reference' for the original configuration."
        ],
        expected_revision: Annotated[str, "The current workspace revision, from read_config or the last edit."],
    ) -> dict[str, str]:
        """Restore a submitted candidate or the reference as the base for further edits."""
        with self._lock:
            key = self.reference_id if candidate_id == "reference" else candidate_id
            if key not in self.snapshots:
                msg = "Unknown candidate ID."
                raise ValueError(msg)
            result = self._write(text=self.snapshots[key], expected_revision=expected_revision)
            self.parent_id = key
            return result

    def _finish(
        self,
        reason: Annotated[
            str,
            "What was considered and why none of it is worth measuring. This ends the experiment with the "
            "remaining evaluations unspent, and is the only record of why.",
        ],
    ) -> str:
        """End optimization when no hypothesis worth measuring remains."""
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

    def tools(self) -> list[Tool]:
        """Build the small tool interface bound to this workspace."""
        return [
            create_tool_from_function(function=self._read_config, name="read_config"),
            create_tool_from_function(function=self._edit_config, name="edit_config"),
            create_tool_from_function(function=self._validate_config, name="validate_config"),
            create_tool_from_function(function=self._submit_candidate, name="submit_candidate"),
            create_tool_from_function(function=self._restore_candidate, name="restore_candidate"),
            create_tool_from_function(function=self._finish, name="finish"),
        ]
