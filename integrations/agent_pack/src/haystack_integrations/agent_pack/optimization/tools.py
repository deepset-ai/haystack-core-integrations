# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from typing import TYPE_CHECKING, Annotated, Any

from haystack.components.agents.state import State
from haystack.core.errors import DeserializationError
from haystack.core.serialization import import_class_by_name
from haystack.lazy_imports import LazyImport
from haystack.tools import tool

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


# The workspace tools change the `ConfigurationWorkspace` passed to `Agent.run` as `workspace`. Each reads it
# from `state.data`, since `State.get` returns a deep copy and the tools have to change the caller's workspace
@tool
def read_config(state: State) -> dict[str, str]:
    """Read the entire editable YAML and its revision for subsequent edits."""
    return state.data["workspace"]._read_config()


@tool
def edit_config(
    state: State,
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
    return state.data["workspace"]._edit_config(old=old, new=new, expected_revision=expected_revision)


@tool
def validate_config(state: State) -> dict[str, Any]:
    """Check the YAML loads as the expected Agent or Pipeline and passes the evaluator's checks, without running."""
    return state.data["workspace"]._validate_config()


@tool
def submit_candidate(
    state: State,
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
    return state.data["workspace"]._submit_candidate(expected_revision=expected_revision, rationale=rationale)


@tool
def restore_candidate(
    state: State,
    candidate_id: Annotated[
        str, "A candidate ID from the outcomes so far, or 'reference' for the original configuration."
    ],
    expected_revision: Annotated[str, "The current workspace revision, from read_config or the last edit."],
) -> dict[str, str]:
    """Restore a submitted candidate or the reference as the base for further edits."""
    return state.data["workspace"]._restore_candidate(candidate_id=candidate_id, expected_revision=expected_revision)


@tool
def finish(
    state: State,
    reason: Annotated[
        str,
        "What was considered and why none of it is worth measuring. This ends the experiment with the "
        "remaining evaluations unspent, and is the only record of why.",
    ],
) -> str:
    """End optimization when no hypothesis worth measuring remains."""
    return state.data["workspace"]._finish(reason=reason)
