# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from importlib.metadata import distributions
from typing import TYPE_CHECKING, Any

from haystack import Pipeline, logging
from haystack import __version__ as haystack_version
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import flatten_tools_or_toolsets, warm_up_tools

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, RunRecord
from haystack_integrations.agent_pack.optimization import prompts
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.workspace import (
    CandidateConfiguration,
    ConfigurationWorkspace,
)
from haystack_integrations.agent_pack.run_digest import (
    EVAL_CASE_SUMMARY_KEY,
    RunDigestPolicy,
    digest_agent_run,
    summarize_eval_case_details,
)

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

logger = logging.getLogger(__name__)


DOCS_SEARCH_TOOL = "search_haystack_docs"

# Documentation the search actually found, per result. Enough for a class signature or a serialization example,
# which is what the optimizer asks this tool for; a whole page is not needed to learn a component's shape.
MAX_DOCUMENTATION_CHARS = 4000


def _documentation_result(payload: Any) -> str:
    """
    Keep the documentation a search found and drop the search engine's own bookkeeping.

    The server answers with its full pipeline debug output, and measured against the live server that is 94% of
    the payload: 183,000 characters of `_debug` around 10,700 characters of documentation. A tool result stays in
    the conversation and is resent on every later step of the turn, so an unfiltered answer costs more context
    than the entire experiment history it is meant to inform.

    :param payload: Whatever the MCP server returned.
    :returns: The retrieved documentation, or the raw answer when it does not have the expected shape.
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
        content = str(document.get("content") or "")[:MAX_DOCUMENTATION_CHARS]
        sections.append(f"[{url}]\n{content}" if url else content)
    return "\n\n".join(sections) or "No documentation matched."


def create_haystack_documentation_mcp_toolset(eager_connect: bool = False) -> "MCPToolset":
    """
    Create the optional read-only public Haystack documentation toolset.

    :param eager_connect: Connect to the documentation server immediately instead of on first use.
    :returns: A toolset exposing Haystack documentation search, reporting only the documentation it found.
    """
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=[DOCS_SEARCH_TOOL],
        eager_connect=eager_connect,
        outputs_to_string={DOCS_SEARCH_TOOL: {"handler": _documentation_result}},
    )


def create_harness_optimizer_agent(
    chat_generator: ChatGenerator | None = None,
    documentation_tools: bool = False,
    system_prompt: str | None = None,
    additional_instructions: str | None = None,
    max_agent_steps: int = 24,
) -> Agent:
    """
    Create the Agent that chooses the next configuration experiment.

    :param chat_generator: Generator used to reason about experiment results and propose configuration changes.
    :param documentation_tools: Give the Agent read-only search over the public Haystack documentation, so it can
        look up a component's parameters and serialized shape rather than guessing at them. Requires
        `mcp-haystack`, and reaches the documentation server over the network.
    :param system_prompt: Optional replacement for the default optimizer instructions. What this environment has
        installed is appended either way, since it constrains every configuration the optimizer can propose.
    :param additional_instructions: Guidance appended to the instructions, for what a good configuration looks like
        in one specific harness. The instructions themselves stay free of any assumption about what the reference
        configuration does, so domain knowledge belongs here rather than in a rewritten replacement.
    :param max_agent_steps: Maximum number of Agent steps used to produce one proposal.
    :returns: The configured optimizer Agent.
    """
    instructions = system_prompt or prompts.HARNESS_OPTIMIZER_SYSTEM_PROMPT
    # Headed like the sections above them, so an appended block reads as its own section rather than as a
    # continuation of whatever the instructions happened to end on.
    instructions = f"{instructions}\n\n## This environment\n\n{describe_environment()}"
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n## This harness\n\n{additional_instructions.strip()}"
    # Measured on the retrieval harness over twenty labelled eval cases: this model reached a better configuration than
    # one costing ten times as much, and a whole experiment on it costs less than a single one of the other's
    # turns. The optimizer's turns are most of what an experiment costs once its eval cases are cheap, so what it is
    # worth paying for them is a question the harness can answer rather than assume.
    generator = chat_generator or OpenAIResponsesChatGenerator(
        model="gpt-5.6-luna",
        timeout=180.0,
        max_retries=5,
        generation_kwargs={"prompt_cache_key": prompts.OPTIMIZER_PROMPT_CACHE_KEY, "reasoning": {"effort": "low"}},
    )
    return Agent(
        chat_generator=generator,
        tools=[create_haystack_documentation_mcp_toolset()] if documentation_tools else None,
        system_prompt=instructions,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )


def describe_environment() -> str:
    """
    Describe what an experiment can actually import, as a line for the optimizer's instructions.

    A configuration is only worth measuring if it can be rebuilt, and whether it can depends on what is installed
    here rather than on what exists. Naming the Haystack version and the integrations present turns a guess about
    availability into a fact the optimizer already has, and a candidate that cannot be constructed costs a whole
    measurement to discover.

    :returns: A sentence naming the Haystack version and every installed Haystack integration.
    """
    integrations = sorted(
        f"{distribution.metadata['Name']} {distribution.version}"
        for distribution in distributions()
        if "haystack" in (distribution.metadata["Name"] or "").lower()
        and distribution.metadata["Name"] != "haystack-ai"
    )
    installed = ", ".join(integrations) if integrations else "none"
    return (
        f"This environment runs Haystack {haystack_version} and has these Haystack integrations installed: "
        f"{installed}. Anything a configuration names has to be importable from those; nothing else is available, "
        f"however well it would suit the experiment."
    )


def _tool_specifications(reference: Agent | Pipeline) -> list[dict[str, Any]]:
    """
    Describe the tools the reference Agent can call.

    A serialized Agent does not reliably carry this: a `ComponentTool` serializes its parameter schema as null
    whenever the schema is derived from the wrapped component, and a `Toolset` that serializes a descriptor of
    itself carries no tool names at all. Without this, the only way to learn what a tool is called and what it
    accepts is to find one already invoked in a recorded run.

    :param reference: The configuration whose tools to describe. A Pipeline that is not an Agent has none.
    :returns: One `{name, description, parameters}` entry per tool, or an empty list when they cannot be read.
    """
    tools = getattr(reference, "tools", None)
    if not tools:
        return []
    try:
        warm_up_tools(tools=tools)
        return [tool.tool_spec for tool in flatten_tools_or_toolsets(tools=tools)]
    except Exception as error:
        logger.warning("Could not describe the reference Agent's tools: {error}", error=error)
        return []


def _fenced(text: str, language: str = "") -> str:
    """
    Wrap text in a fence long enough that nothing inside can close it early.

    :param text: Content to fence, reproduced exactly.
    :param language: Optional language hint for the opening fence.
    :returns: The fenced block.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _section(title: str, body: str) -> str:
    """Render one titled section, or nothing when there is no body."""
    return f"## {title}\n\n{body}\n\n" if body else ""


def _headline(metrics: dict[str, Any] | None) -> str:
    """
    Reduce one outcome's measurement to what a reader compares across candidates.

    An earlier candidate is read to decide what to try next, and that decision turns on how it scored and what it
    failed, not on its token counts. The most recent outcome is sent separately in full.

    :param metrics: The candidate's measurement, or None when it could not be measured.
    :returns: A single line of headline numbers.
    """
    if metrics is None:
        return "not measured"
    details = metrics.get("details") or {}
    parts = [f"quality {metrics.get('quality'):.3f}" if metrics.get("quality") is not None else "quality unknown"]
    cost = metrics.get("cost")
    parts.append("cost unpriced" if cost is None else f"cost ${cost:.4f}")
    for key in (
        "mean_recall",
        "mean_precision",
        "mean_recall_at_k",
        "mean_precision_at_k",
        "mean_retrieved",
        "mean_queries",
    ):
        if key in details:
            parts.append(f"{key.removeprefix('mean_')} {details[key]:.2f}")
    # In execution order, so the arrow shows where the path widened and where it narrowed again.
    if stages := details.get("mean_stage_outputs"):
        parts.append(
            "stages "
            + " -> ".join(
                f"{component}.{socket} {size:.1f}"
                for component, sockets in stages.items()
                for socket, size in sockets.items()
            )
        )
    if (summary := details.get(EVAL_CASE_SUMMARY_KEY)) is not None:
        parts.append(f"{summary.get('passed')}/{summary.get('eval_cases')} eval cases clean")
        if failures := summary.get("failures"):
            parts.append("failures " + ", ".join(f"{name} x{count}" for name, count in failures.items()))
    if details.get("usage_complete") is False:
        parts.append("usage incomplete")
    for warning in details.get("warnings") or []:
        parts.append(f"warning x{warning['count']}: {warning['message']}")
    return " | ".join(parts)


def _render_outcomes(history: list[dict[str, Any]]) -> str:
    """
    Render every measured candidate as a readable entry, newest last.

    :param history: Prior candidate outcomes.
    :returns: One section per candidate.
    """
    entries = []
    for position, entry in enumerate(history, start=1):
        parent = (entry.get("parent_id") or "reference")[:12]
        gates = ", ".join(entry.get("gate_failures") or ()) or "passed"
        lines = [f"### {position}. `{entry['candidate_id'][:12]}` from `{parent}` — gates {gates}"]
        lines.append(_headline(metrics=entry.get("metrics")))
        if failure := entry.get("failure"):
            lines.append(f"failed to evaluate: {failure}")
        if rationale := entry.get("rationale"):
            lines.append(f"hypothesis: {rationale}")
        if diff := entry.get("diff"):
            lines.append(_fenced(text=diff.rstrip(), language="diff"))
        entries.append("\n\n".join(lines))
    return "\n\n".join(entries)


def propose_candidate(
    optimizer_agent: Agent,
    workspace: ConfigurationWorkspace,
    reference: Agent | Pipeline,
    reference_runs: list[RunRecord],
    pricing: ModelPriceCatalog,
    objectives: OptimizationObjectives,
    baseline: EvaluationMetrics,
    history: list[dict[str, Any]],
    digest_policy: RunDigestPolicy | None = None,
    history_digest_window: int = 1,
    remaining_evaluations: int | None = None,
    base_id: str | None = None,
) -> CandidateConfiguration | None:
    """
    Let the optimizer edit, validate and submit one YAML candidate.

    :param optimizer_agent: Agent supplying generator, instructions and optional documentation tools.
    :param workspace: Single editable file and previous snapshots.
    :param reference: Reference configuration supplying tool specifications, when it has any.
    :param reference_runs: Recorded behavior evidence.
    :param pricing: Known model prices.
    :param objectives: Quality gates and ranking objective.
    :param baseline: Reference measurement.
    :param history: Prior candidate outcomes.
    :param digest_policy: Run evidence limits.
    :param history_digest_window: Number of recent detailed outcomes.
    :param remaining_evaluations: How many candidates, including this one, the experiment can still measure. A
        submission costs one pass over the whole evaluation set, so without this the optimizer cannot tell a
        measurement it can afford to spend on one small change from its last remaining one.
    :param base_id: Candidate this turn's edits start from. Defaults to whatever was submitted last.
    :returns: Submitted snapshot, or None after finish or exhaustion of the proposal step budget.
    """
    workspace.begin_turn(base_id=base_id)
    configuration = workspace.read_config()

    # The three messages are ordered by how they change, so that as much of each turn as possible is a cache hit:
    # caching matches the longest common token prefix, so anything rewritten has to come after anything that is not.

    # First message: identical every turn, so it caches in full.
    tools = _tool_specifications(reference=reference)
    context = "".join(
        [
            _section(title="Objectives", body=json.dumps(objectives.to_dict())),
            _section(title="Known model prices", body=json.dumps(pricing.to_dict())),
            _section(title="Tools available to the reference", body=json.dumps(tools) if tools else ""),
            _section(
                title="How the reference behaved",
                body="\n\n".join(
                    _fenced(text=json.dumps(digest_agent_run(result=record.outputs, policy=digest_policy), default=str))
                    for record in reference_runs[:3]
                    if record.outputs
                ),
            ),
            # Described eval case by eval case only while it is the only thing measured; once a candidate exists, the
            # most recent one is the configuration worth reading in that much detail.
            _section(
                title="Reference measurement",
                body=json.dumps(
                    baseline.to_dict() if not history else summarize_eval_case_details(payload=baseline.to_dict())
                ),
            ),
        ]
    )

    # Second message: append-only, so every turn re-reads all but the newest entry from cache.
    outcomes = _section(
        title="Outcomes so far", body=_render_outcomes(history=summarize_eval_case_details(payload=history))
    )

    # Third message: rewritten every turn, so none of it is cacheable and all of it goes last.
    recent = history[-history_digest_window:] if history_digest_window > 0 else []
    current = "".join(
        [
            _section(
                title="Budget",
                body="unknown"
                if remaining_evaluations is None
                else f"{remaining_evaluations} evaluations remain, including this one.",
            ),
            _section(
                title="Most recent outcome in full",
                body="\n\n".join(json.dumps(entry, default=str) for entry in recent),
            ),
            _section(
                title="candidate.yaml",
                body=f"revision `{configuration['revision']}`, edited from `{configuration['parent_id'][:12]}`\n\n"
                # Reproduced as a fenced block rather than a JSON string: this is the text the editing tools match
                # against, and a JSON string would show it with its newlines and quotes escaped.
                + _fenced(text=configuration["yaml"], language="yaml"),
            ),
        ]
    )

    # We must create a new Agent for each turn, because the tools are bound to the workspace and the workspace is bound
    # to the turn. So we clone the optimizer_agent and add the workspace tools to it and update its exit conditions.
    agent = optimizer_agent.clone(
        tools=[*optimizer_agent.tools, *workspace.tools()],
        tool_concurrency_limit=1,
        exit_conditions=["submit_candidate", "finish"],
    )
    result = agent.run(
        messages=[
            ChatMessage.from_user(text=context),
            ChatMessage.from_user(text=outcomes),
            ChatMessage.from_user(text=current),
        ]
    )

    logger.info(
        "optimizer turn: steps={steps}/{budget} exit={exit_reason} calls={calls} usage={usage}",
        steps=result.get("step_count"),
        budget=agent.max_agent_steps,
        exit_reason=result.get("exit_reason"),
        calls=[call.tool_name for message in result.get("messages") or [] for call in message.tool_calls],
        usage=result.get("token_usage"),
    )
    return workspace.submitted
