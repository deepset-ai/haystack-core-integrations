# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import yaml
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.core.errors import DeserializationError
from haystack.core.serialization import import_class_by_name
from haystack.marshal import YamlMarshaller
from haystack.tools import Tool, flatten_tools_or_toolsets
from haystack.tools.from_function import create_tool_from_function


class _ReadableDumper(yaml.SafeDumper):
    pass


def content_digest(payload: str) -> str:
    """
    Return a short, stable digest of serialized content.

    :param payload: The serialized content to identify.
    :returns: A twelve-character hexadecimal digest.
    """
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_ReadableDumper.add_representer(str, _represent_string)


class _ReadableYamlMarshaller(YamlMarshaller):
    def marshal(self, dict_: dict[str, Any]) -> str:
        """Keep multiline prompts editable without escaped newlines or Unicode."""
        return yaml.dump(dict_, Dumper=_ReadableDumper, allow_unicode=True, width=120)


def dump_pipeline(pipeline: Pipeline) -> str:
    """Emit readable Haystack Pipeline YAML."""
    return pipeline.dumps(marshaller=_ReadableYamlMarshaller())


def dump_agent(agent: Agent) -> str:
    """Wrap an independent copy of the Agent and emit readable Haystack Pipeline YAML."""
    pipeline = Pipeline()
    pipeline.add_component("agent", agent.clone())
    return dump_pipeline(pipeline=pipeline)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Reject duplicate keys instead of silently discarding an earlier edit."""
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise yaml.constructor.ConstructorError(None, None, "Duplicate YAML key", node.start_mark)
        return super().construct_mapping(node, deep=deep)


def configuration_id(text: str) -> str:
    """Hash parsed configuration, retaining resource identities but ignoring YAML formatting."""
    # _UniqueKeyLoader subclasses SafeLoader; it only adds duplicate-key rejection.
    return content_digest(json.dumps(yaml.load(text, Loader=_UniqueKeyLoader), sort_keys=True))  # noqa: S506


def load_pipeline(text: str) -> Pipeline:
    """Load a Pipeline using Haystack's deserialization security, rejecting duplicate keys first."""
    # Parsed twice on purpose: Haystack's own loader accepts a repeated key and keeps the last one, which would
    # silently discard an edit the optimizer believes it made.
    yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    return Pipeline.loads(text)


def load_agent(text: str) -> Agent:
    """Load an Agent from its one-component Pipeline using Haystack's deserialization security."""
    data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    if not isinstance(data, dict) or set(data.get("components", {})) != {"agent"} or data.get("connections"):
        msg = "The outer Pipeline must contain exactly one component named 'agent' and no connections."
        raise ValueError(msg)
    agent = load_pipeline(text).get_component("agent")
    if not isinstance(agent, Agent):
        msg = "The 'agent' component must be a Haystack Agent."
        raise ValueError(msg)
    return agent


@dataclass
class CandidateConfiguration:
    """A complete submitted configuration, independent of subsequent workspace edits."""

    candidate_id: str
    parent_id: str
    yaml: str
    rationale: str
    diff: str


class ConfigurationWorkspace:
    """Tools are bound to one file; snapshots and the reference are owned by the runner."""

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
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(reference_yaml)
        self._read()
        self.loader = loader
        self.reference_id = configuration_id(reference_yaml)
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
        if self.path.is_symlink():
            msg = "The configuration file must not be a symlink."
            raise ValueError(msg)
        return self.path.read_text(encoding="utf-8")

    def _write(self, text: str, expected_revision: str) -> dict[str, str]:
        if self.submitted is not None or self.finished:
            msg = "This proposal turn has ended."
            raise ValueError(msg)
        if content_digest(self._read()) != expected_revision:
            msg = "Stale revision: read_config again before editing."
            raise ValueError(msg)
        self._overwrite(text)
        self.validated_revision = None
        return {"revision": content_digest(text)}

    def _overwrite(self, text: str) -> None:
        """Replace the directory entry atomically, rather than following a possible file symlink on write."""
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, self.path)

    def read_config(self) -> dict[str, str]:
        """Read the entire editable YAML and its revision for subsequent edits."""
        with self._lock:
            text = self._read()
            return {"yaml": text, "revision": content_digest(text), "parent_id": self.parent_id}

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
            text = self._read()
            if not old or text.count(old) != 1:
                msg = "old must match exactly once; include more surrounding text to disambiguate."
                raise ValueError(msg)
            return self._write(text.replace(old, new, 1), expected_revision)

    def validate_config(self) -> dict[str, Any]:
        """Check YAML, deserialization, and evaluator contracts without running or warming the Agent."""
        with self._lock:
            text = self._read()
            revision = content_digest(text)
            self.validated_revision = None
            try:
                loaded = self.loader(text)
                try:
                    if self.validator is not None:
                        self.validator(loaded)
                    # A Pipeline that is not an Agent has no tools, and reporting an empty list is the honest
                    # answer rather than a missing key the reader has to interpret.
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
            text = self._read()
            if self.finished or self.submitted is not None:
                msg = "This proposal turn has ended."
                raise ValueError(msg)
            if expected_revision != content_digest(text) or self.validated_revision != expected_revision:
                msg = "Validate the current revision before submitting."
                raise ValueError(msg)
            candidate_id = configuration_id(text)
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
            self.submitted = CandidateConfiguration(candidate_id, self.parent_id, text, rationale, diff)
            self.snapshots[candidate_id] = text
            return {"candidate_id": candidate_id}

    def restore_candidate(
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
            result = self._write(self.snapshots[key], expected_revision)
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

    def begin_turn(self, base_id: str | None = None) -> None:
        """
        Continue from a chosen base, retaining every restore point.

        :param base_id: Snapshot the next edits start from. Without one the turn continues from whatever was
            submitted last, which makes a run that regresses carry the regression into everything after it: each
            edit builds on the previous attempt rather than on the best one, so a search can spend its budget
            walking away from a configuration it already found. Passing the best measured candidate makes the
            default a climb instead.
        """
        with self._lock:
            if base_id is not None and base_id in self.snapshots:
                if self._read() != self.snapshots[base_id]:
                    self._overwrite(self.snapshots[base_id])
                self.parent_id = base_id
            elif self.submitted is not None:
                self.parent_id = self.submitted.candidate_id
            self.submitted = None
            self.validated_revision = None

    def tools(self) -> list[Tool]:
        """Build the small tool interface bound to this workspace."""
        return [
            create_tool_from_function(function=method)
            for method in (
                self.read_config,
                self.edit_config,
                self.validate_config,
                self.submit_candidate,
                self.restore_candidate,
                self.finish,
                inspect_component,
            )
        ]


def _installed_path(type_name: str) -> str:
    """
    Try to find the import path of a class, given a plausible but wrong path to it.

    :param type_name: A fully qualified class name that could not be imported.
    :returns: The path the class is installed at, or the original name when nothing of that name is installed.
    """
    parts = type_name.split(".")
    for cut in range(len(parts) - 2, 0, -1):
        try:
            # We make sure to use import_class_by_name to respect the deserialization allowlist
            found = import_class_by_name(".".join([*parts[:cut], parts[-1]]))
        except (DeserializationError, ImportError):
            continue
        return f"{found.__module__}.{found.__qualname__}"
    return type_name


def inspect_component(
    type_name: Annotated[
        str,
        "Fully qualified class name, as written in a configuration's 'type' field, for example "
        "'haystack.components.rankers.llm_ranker.LLMRanker'. A path that does not import is retried where the "
        "class is installed, and the answer reports the path to use.",
    ],
) -> dict[str, str]:
    """Inspect an installed class using the same namespace allowlist as deserialization."""
    try:
        cls = import_class_by_name(type_name)
    except ImportError:
        type_name = _installed_path(type_name)
        cls = import_class_by_name(type_name)
    return {
        "import_path": type_name,
        "constructor": str(inspect.signature(cls.__init__)),
        "documentation": inspect.getdoc(cls) or "",
        "run": str(inspect.signature(cls.run)) if hasattr(cls, "run") else "",
        "run_documentation": (inspect.getdoc(cls.run) or "") if hasattr(cls, "run") else "",
        "serialization": inspect.getsource(cls.to_dict)
        if hasattr(cls, "to_dict")
        else "Default Haystack serialization",
    }
