# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from difflib import unified_diff
from threading import RLock
from typing import Annotated, Any

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.tools import Toolset, create_tool_from_function, flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.dataclasses import CandidateConfiguration
from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    content_digest,
    load_agent,
)


class ConfigurationEditor(Toolset):
    """
    The configuration YAML the optimizer edits during one proposal turn, and the tools it edits it with.

    Its tools are `read_config`, `edit_config`, `validate_config`, `submit_candidate`, `restore_candidate` and
    `finish`. Pass it in the optimizer agent's `tools` for one run. The turn ends when the optimizer submits, which
    puts the YAML into `submitted` as a `CandidateConfiguration`, or when it calls `finish`, which records
    `finish_reason`. A new editor is created for every turn.
    """

    def __init__(
        self,
        *,
        reference_yaml: str,
        candidates: dict[str, str] | None = None,
        base_id: str | None = None,
        validator: Callable[[Agent | Pipeline], None] | None = None,
        loader: Callable[[str], Agent | Pipeline] = load_agent,
    ) -> None:
        """
        Start a turn from the reference or from an earlier candidate.

        :param reference_yaml: The reference configuration, as one Pipeline YAML document. The optimizer can restore
            it as `"reference"`.
        :param candidates: The YAML of every candidate submitted on earlier turns, keyed by candidate ID. The
            optimizer can restore any of them, and submitting one of them again is refused.
        :param base_id: The candidate this turn's edits start from, or `None` to start from the reference.
        :param validator: Optional evaluator-specific check after deserialization.
        :param loader: Builds the configuration from YAML, and decides what shape a candidate must keep. Defaults
            to the Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.
        :raises ValueError: If `base_id` is not one of `candidates`.
        """
        self.reference_id = _configuration_id(reference_yaml)
        # Every configuration the optimizer can restore, the reference included
        self.configurations = {self.reference_id: reference_yaml, **(candidates or {})}
        if base_id is not None and base_id not in self.configurations:
            msg = f"Unknown base candidate ID: {base_id}."
            raise ValueError(msg)
        self.parent_id = base_id or self.reference_id
        self.text = self.configurations[self.parent_id]
        self.validator = validator
        self.loader = loader
        self.validated_revision: str | None = None
        self.submitted: CandidateConfiguration | None = None
        self.finished = False
        self.finish_reason: str | None = None
        self.validation_failures: list[dict[str, str]] = []
        self._lock = RLock()
        # Each tool is a method of this editor, so it reads and changes this turn's YAML directly
        super().__init__(
            tools=[
                create_tool_from_function(function=method)
                for method in (
                    self.read_config,
                    self.edit_config,
                    self.validate_config,
                    self.submit_candidate,
                    self.restore_candidate,
                    self.finish,
                )
            ]
        )

    def _write(self, text: str, expected_revision: str) -> dict[str, str]:
        """Replace the YAML if the turn is open and `expected_revision` is current, and return the new revision."""
        if self.submitted is not None or self.finished:
            msg = "This proposal turn has ended."
            raise ValueError(msg)
        if content_digest(payload=self.text) != expected_revision:
            msg = "Stale revision: read_config again before editing."
            raise ValueError(msg)
        self.text = text
        self.validated_revision = None
        return {"revision": content_digest(payload=text)}

    # The docstrings and `Annotated` descriptions below are what the optimizer reads as each tool's description.

    def read_config(self) -> dict[str, str]:
        """Read the entire editable YAML and its revision for subsequent edits."""
        with self._lock:
            return {"yaml": self.text, "revision": content_digest(payload=self.text), "parent_id": self.parent_id}

    def edit_config(
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
            if not old or self.text.count(old) != 1:
                msg = "old must match exactly once; include more surrounding text to disambiguate."
                raise ValueError(msg)
            return self._write(text=self.text.replace(old, new, 1), expected_revision=expected_revision)

    def validate_config(self) -> dict[str, Any]:
        """Check the YAML loads as the expected Agent or Pipeline and passes the evaluator's checks, without running."""
        with self._lock:
            revision = content_digest(payload=self.text)
            self.validated_revision = None
            try:
                loaded = self.loader(self.text)
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

    def submit_candidate(
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
            if self.finished or self.submitted is not None:
                msg = "This proposal turn has ended."
                raise ValueError(msg)
            if expected_revision != content_digest(payload=self.text) or self.validated_revision != expected_revision:
                msg = "Validate the current revision before submitting."
                raise ValueError(msg)
            candidate_id = _configuration_id(self.text)
            if candidate_id in self.configurations:
                msg = "duplicate_or_no_op: this configuration was already submitted or is the reference."
                raise ValueError(msg)
            diff = "".join(
                unified_diff(
                    self.configurations[self.parent_id].splitlines(keepends=True),
                    self.text.splitlines(keepends=True),
                    fromfile=self.parent_id,
                    tofile=candidate_id,
                )
            )
            self.submitted = CandidateConfiguration(
                candidate_id=candidate_id, parent_id=self.parent_id, yaml=self.text, rationale=rationale, diff=diff
            )
            return {"candidate_id": candidate_id}

    def restore_candidate(
        self,
        candidate_id: Annotated[
            str, "A candidate ID from the outcomes so far, or 'reference' for the original configuration."
        ],
        expected_revision: Annotated[str, "The current revision, from read_config or the last edit."],
    ) -> dict[str, str]:
        """Restore a submitted candidate or the reference as the base for further edits."""
        with self._lock:
            key = self.reference_id if candidate_id == "reference" else candidate_id
            if key not in self.configurations:
                msg = "Unknown candidate ID."
                raise ValueError(msg)
            result = self._write(text=self.configurations[key], expected_revision=expected_revision)
            self.parent_id = key
            return result

    def finish(
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
