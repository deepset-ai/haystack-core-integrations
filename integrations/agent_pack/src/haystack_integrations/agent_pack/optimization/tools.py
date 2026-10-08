# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from difflib import unified_diff
from typing import TYPE_CHECKING, Annotated, Any, cast

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.core.errors import DeserializationError
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool, flatten_tools_or_toolsets, tool
from haystack.utils import deserialize_callable, serialize_callable

from haystack_integrations.agent_pack.evaluation.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.optimization.dataclasses import (
    CandidateConfiguration,
    ConfigurationDraft,
    KnownConfigurations,
)
from haystack_integrations.agent_pack.optimization.utils import _configuration_id, load_agent

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Run 'pip install mcp-haystack'") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo


def _documentation_result(payload: Any) -> str:
    """
    Keep only the `documents` key of a documentation search answer and drop every other key, such as `_debug`.

    :param payload: Whatever the MCP server returned.
    :returns: Each document's content, prefixed with its URL when it has one. The raw answer is returned when it
        has no `documents` key.
    """
    try:
        # The server answers inside an MCP envelope, which arrives already serialized, so the body is reached by
        # parsing twice: once for the envelope and once for the payload its single text content carries.
        body = payload if isinstance(payload, dict) else json.loads(str(payload))
        if "documents" not in body:
            body = json.loads(body["content"][0]["text"])
        documents = body["documents"]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return str(payload)
    sections = []
    for document in documents:
        url = (document.get("meta") or {}).get("url", "")
        content = document.get("content") or ""
        sections.append(f"[{url}]\n{content}" if url else content)
    return "\n\n".join(sections) or "No documentation matched."


def _make_haystack_documentation_toolset() -> "MCPToolset":
    """Create the optional read-only public Haystack documentation toolset."""
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=["search_haystack_docs"],
        outputs_to_string={"search_haystack_docs": {"handler": _documentation_result}},
    )


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
            found = import_class_by_name(fully_qualified_name=".".join([*parts[:cut], parts[-1]]))
        except (DeserializationError, ImportError):
            continue
        return f"{found.__module__}.{found.__qualname__}"
    return type_name


@tool
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
        cls = import_class_by_name(fully_qualified_name=type_name)
    except ImportError:
        type_name = _installed_path(type_name=type_name)
        cls = import_class_by_name(fully_qualified_name=type_name)
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


def _check_revision(draft: ConfigurationDraft, expected_revision: str) -> None:
    """
    Refuse an edit made against a revision that is no longer current.

    :param draft: The YAML being edited.
    :param expected_revision: The revision the optimizer named.
    :raises ValueError: If `expected_revision` is not the draft's current revision.
    """
    if draft.revision != expected_revision:
        msg = "Stale revision: read_config again before editing."
        raise ValueError(msg)


# The editing tools share one proposal turn's state: `draft` is the YAML being edited, `known` holds every
# configuration measured before the turn, and `submitted`, `finish_reason` and `validation_failures` are what the turn
# produced. Each tool declares the keys it reads and writes, and returns a new `draft` rather than changing it.


@tool(inputs_from_state={"draft": "draft"})
def read_config(draft: ConfigurationDraft) -> dict[str, str]:
    """Read the entire editable YAML and its revision for subsequent edits."""
    return {"yaml": draft.yaml, "revision": draft.revision, "parent_id": draft.parent_id}


@tool(
    inputs_from_state={"draft": "draft"},
    outputs_to_state={"draft": {"source": "draft"}},
    outputs_to_string={"source": "revision"},
)
def edit_config(
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
    draft: ConfigurationDraft,
) -> dict[str, Any]:
    """Replace one exact text block. Use the entire current YAML as old for a full rewrite."""
    _check_revision(draft=draft, expected_revision=expected_revision)
    if not old or draft.yaml.count(old) != 1:
        msg = "old must match exactly once; include more surrounding text to disambiguate."
        raise ValueError(msg)
    # Any edit clears the validation, so the new YAML has to be validated again before it can be submitted
    edited = replace(draft, yaml=draft.yaml.replace(old, new, 1), validated_revision=None)
    return {"draft": edited, "revision": edited.revision}


class ValidateConfig(Tool):
    """The `validate_config` tool, holding the loader and the evaluator whose check a candidate has to pass."""

    def __init__(self, *, evaluator: HarnessEvaluator, loader: Callable[[str], Agent | Pipeline] = load_agent) -> None:
        """
        Create the tool.

        :param evaluator: The evaluator the candidates are measured with. Its `validate` runs on every loaded
            configuration, and its `to_dict` and `from_dict` let this tool, and the agent holding it, be serialized.
        :param loader: Builds the configuration from YAML, and decides what shape a candidate must keep. Defaults
            to the Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.
        :raises TypeError: If the evaluator is missing `validate`, `to_dict` or `from_dict`.
        """
        missing = [
            name for name in ("validate", "to_dict", "from_dict") if not callable(getattr(evaluator, name, None))
        ]
        if missing:
            msg = f"The evaluator must implement {', '.join(missing)} to validate and serialize candidates."
            raise TypeError(msg)
        self.evaluator = evaluator
        self.loader = loader
        super().__init__(
            name="validate_config",
            description=(
                "Check the YAML loads as the expected Agent or Pipeline and passes the evaluator's checks, without "
                "running."
            ),
            parameters={"type": "object", "properties": {}},
            function=self._validate,
            inputs_from_state={"draft": "draft"},
            outputs_to_state={
                "draft": {"source": "draft"},
                "validation_failures": {"source": "validation_failures"},
            },
            outputs_to_string={"source": "result"},
        )

    def _validate(self, draft: ConfigurationDraft) -> dict[str, Any]:
        """
        Load the YAML and run the evaluator's check on it.

        :param draft: The YAML to check.
        :returns: What the optimizer is shown under `result`, the draft with `validated_revision` set when both
            passed, and the failure under `validation_failures` when they did not.
        """
        try:
            loaded = self.loader(draft.yaml)
            try:
                self.evaluator.validate(loaded)
                # A Pipeline that is not an Agent has no tools
                specs = [item.tool_spec for item in flatten_tools_or_toolsets(getattr(loaded, "tools", None) or [])]
            finally:
                loaded.close()
        except Exception as error:
            failure = {"revision": draft.revision, "error": f"{type(error).__name__}: {error}"}
            return {
                "result": {"valid": False, **failure},
                "draft": replace(draft, validated_revision=None),
                "validation_failures": [failure],
            }
        return {
            "result": {"valid": True, "revision": draft.revision, "tools": specs},
            "draft": replace(draft, validated_revision=draft.revision),
            "validation_failures": [],
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tool.

        :returns: The evaluator as its own `to_dict` describes it, and the loader as an import path.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"evaluator": self.evaluator.to_dict(), "loader": serialize_callable(self.loader)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidateConfig":
        """
        Deserialize the tool.

        :param data: What `to_dict` returned.
        :returns: The restored tool.
        """
        evaluator_data = data["data"]["evaluator"]
        evaluator_class = cast(
            type[HarnessEvaluator], import_class_by_name(fully_qualified_name=evaluator_data["type"])
        )
        evaluator = evaluator_class.from_dict(evaluator_data)
        return cls(evaluator=evaluator, loader=deserialize_callable(data["data"]["loader"]))


@tool(
    inputs_from_state={"draft": "draft", "known": "known"},
    outputs_to_state={"submitted": {"source": "submitted"}},
    outputs_to_string={"source": "candidate_id"},
)
def submit_candidate(
    expected_revision: Annotated[
        str, "The revision returned by a successful validate_config, which must still be current."
    ],
    rationale: Annotated[
        str,
        "The hypothesis this candidate tests: what was changed and what it is expected to move. Read back "
        "alongside the score, so name the change rather than restating the goal.",
    ],
    draft: ConfigurationDraft,
    known: KnownConfigurations,
) -> dict[str, Any]:
    """Submit this validated revision for evaluation and end the proposal turn."""
    if expected_revision != draft.revision or draft.validated_revision != expected_revision:
        msg = "Validate the current revision before submitting."
        raise ValueError(msg)
    candidate_id = _configuration_id(draft.yaml)
    if candidate_id in known.yaml_by_id:
        msg = "duplicate_or_no_op: this configuration was already submitted or is the reference."
        raise ValueError(msg)
    diff = "".join(
        unified_diff(
            known.yaml_by_id[draft.parent_id].splitlines(keepends=True),
            draft.yaml.splitlines(keepends=True),
            fromfile=draft.parent_id,
            tofile=candidate_id,
        )
    )
    submitted = CandidateConfiguration(
        candidate_id=candidate_id, parent_id=draft.parent_id, yaml=draft.yaml, rationale=rationale, diff=diff
    )
    return {"candidate_id": candidate_id, "submitted": submitted}


@tool(
    inputs_from_state={"draft": "draft", "known": "known"},
    outputs_to_state={"draft": {"source": "draft"}},
    outputs_to_string={"source": "revision"},
)
def restore_candidate(
    candidate_id: Annotated[
        str, "A candidate ID from the outcomes so far, or 'reference' for the original configuration."
    ],
    expected_revision: Annotated[str, "The current revision, from read_config or the last edit."],
    draft: ConfigurationDraft,
    known: KnownConfigurations,
) -> dict[str, Any]:
    """Restore a submitted candidate or the reference as the base for further edits."""
    _check_revision(draft=draft, expected_revision=expected_revision)
    key = known.reference_id if candidate_id == "reference" else candidate_id
    if key not in known.yaml_by_id:
        msg = "Unknown candidate ID."
        raise ValueError(msg)
    restored = ConfigurationDraft(yaml=known.yaml_by_id[key], parent_id=key)
    return {"draft": restored, "revision": restored.revision}


@tool(outputs_to_state={"finish_reason": {"source": "finish_reason"}}, outputs_to_string={"source": "message"})
def finish(
    reason: Annotated[
        str,
        "What was considered and why none of it is worth measuring. This ends the experiment with the "
        "remaining evaluations unspent, and is the only record of why.",
    ],
) -> dict[str, str]:
    """End optimization when no hypothesis worth measuring remains."""
    return {"message": "Finished.", "finish_reason": reason}
