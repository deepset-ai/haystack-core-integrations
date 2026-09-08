# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The Agent that chooses optimization experiments and the requests made to it."""

import json
import re
from importlib.metadata import distributions
from typing import TYPE_CHECKING, Any

from haystack import __version__ as haystack_version
from haystack import logging
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset, flatten_tools_or_toolsets, warm_up_tools

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.workspace import (
    CandidateConfiguration,
    ConfigurationWorkspace,
    Optimizable,
)
from haystack_integrations.agent_pack.run_digest import (
    CASE_SUMMARY_KEY,
    RunDigestPolicy,
    digest_agent_run,
    summarize_case_details,
)

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

logger = logging.getLogger(__name__)

# Version the routing key with the request layout: a changed layout is a different reusable prefix.
OPTIMIZER_PROMPT_CACHE_KEY = "haystack-harness-optimizer-yaml-v1"

HARNESS_OPTIMIZER_SYSTEM_PROMPT = """
Optimize a Haystack pipeline through measured experiments. Each turn you edit its YAML and submit the result to be
measured against a fixed evaluation set.

## The turn

1. Start from the best candidate measured so far, or restore_candidate to a different base deliberately.
2. Form one hypothesis about what would improve the configuration.
3. Edit candidate.yaml to test it.
4. Validate, and repair any errors.
5. Submit with a rationale.
6. Read the score and run evidence to choose the next hypothesis.

The rest of these instructions are why each step goes the way it does.

## The workspace

The configuration is in candidate.yaml, a serialized Haystack Pipeline. What it holds varies with the experiment:
connected components, or a single component that is an Agent, which is how an Agent is serialized. Read it
rather than assuming either. Use read_config and edit_config to edit YAML directly. You may change prompts,
models, parameters, hooks, and add, remove or replace entire tools and pipelines. Each edit requires the latest
revision and a unique exact text match. The tools can only edit that file.

Use validate_config and repair errors before submit_candidate. Validation constructs the pipeline but does not
run or warm it up. Submit one hypothesis per turn with a rationale. Plain text does not submit a candidate. Invalid
drafts and duplicates do not spend evaluation slots, but editing steps are bounded.

## Spending the budget

Spend the evaluations. `remaining_evaluations` is a budget rather than a limit to stay under, and one left unused
is a measurement not taken. Do not stop merely because one experiment regresses. When the obvious parameters have
been tried, the configuration is still open: a component's prompt, what a component is asked to produce rather than 
how much, the shape of the pipeline, and components not yet in it. Reach for finish only when you can say what you 
considered and why none of it is worth measuring. It takes that reason as an argument and ends the experiment.

Combine changes when they need to move together, and combine the change you are measuring with cleanups that
cannot plausibly interact with it: `remaining_evaluations` counts submissions and each one costs a full pass over
the evaluation set. Removing an unused tool also removes its schema from model input.

## Choosing the next experiment

Each turn starts from the best candidate measured so far, not from whatever was tried last, so a variation that
regresses is not inherited by the next one and you are always varying against the best known configuration. Vary
one thing at a time against it. Edits continue from that base; use restore_candidate with a history ID or
'reference' to leave it deliberately rather than to climb back to it: to retry a structural change whose
parameters were wrong rather than its shape, or to return to 'reference' and take a different direction entirely
when a line of variations has stopped paying.

Varying one thing at a time says how to change the configuration, not which thing to keep changing. When
successive candidates move the same component in the same direction and the scores follow in order, that axis is
answered, and the next point on it will restate what the earlier ones showed. A monotone series against one
component is the signal to go and measure a different one: most configurations have components that have been
measured once, and the one measured five times is rarely where the remaining quality is.

A component that selects or filters what it is handed saturates in a particular way. Once it is set to keep
everything it can, the measurement stops describing that component and starts describing its input, so further
variations of it land on the same score — what is missing was never in front of it. Repeated ties at the
permissive end of such a component mean the limit is upstream of it.

## Where quality usually hides

A component's default prompt is part of the configuration and is one of the most productive things to change. It
was written for that component's general case, not for what is being measured here, and a default that quietly
mismatches the task costs quality without ever failing: read the prompt in the YAML before assuming it fits.

Read a limit against what the run actually did with it. A component producing less than its own limit allows is
leaving that room unspent, and the reason is usually in its prompt rather than in the number. A limit reached on
every case is the opposite: it is binding, and what it truncates is invisible until it is raised.

## OpenAI generators

Every OpenAI generator in a configuration should be `OpenAIResponsesChatGenerator`, including one held inside
another component. Haystack reaches OpenAI two ways, and the choice decides what the model can do: the responses
endpoint supports reasoning, the completions endpoint behind `OpenAIChatGenerator` does not, and a current model
placed there simply reasons less without anything failing. Choose it when adding a component rather than copying
whichever class the surrounding configuration or a usage example happened to use, and change the ones already
there: a reference written against the completions generator is getting less out of its model, and swapping
the class is a small edit with nothing else riding on it.

Its parameters go in `generation_kwargs`, and anything `openai.Responses.create` accepts works:

- reasoning depth: `{"reasoning": {"effort": "low" | "medium" | "high"}}`, optionally with `"summary": "auto"`
- structured output: `{"text": {"format": {"type": "json_schema", "name": ..., "strict": true, "schema": {...}}}}`
- `temperature` is rejected outright by the current models, and `text_format` takes a Pydantic class that no
  serialized configuration can carry, so the JSON schema goes in `text` as above

Constrain the reply with that schema wherever a component parses it rather than passing it on — a ranker reading
indices, an expander reading a list. Such a component does not recover: it catches the parse error, logs a
warning, and returns something unranked or unexpanded, so the run continues and the measurement describes the
fallback rather than the configuration. Asking for well-formed output in the prompt is not the same as requiring
it, and what a component falls back to is worth knowing: one returns the documents unranked, another returns the
single unexpanded query, and both look like an ordinary poor score.

Both of these are properties of every generator in the configuration, so audit them once at the start rather than
finding them one failure at a time. A reference is usually wrong about them uniformly — the same class copied into
each component, no schema on any of them — and fixing them together costs one measurement instead of several.

## Learning what is available

Use inspect_component and optional documentation tools to learn installed components and their serialization.
A ComponentTool can become a PipelineTool: connect retriever.documents to ranker.documents, map query to both query
inputs, filters to the retriever, and ranker.documents to the tool documents output. Preserve outputs_to_state and
formatting handlers required by the harness. Do not invent serialization shapes or assume a package is installed.

## How a candidate is judged

Quality is a hard gate. Known prices are informational rather than an allowlist. Unpriced or incomplete usage
cannot win a cost optimization. Read gate failures and run evidence before choosing the next experiment.
Run evidence is compressed: truncated results and incomplete listings are explicitly marked. Do not read a mark
as evidence that nothing happened there.
""".strip()


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
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    additional_instructions: str | None = None,
    max_agent_steps: int = 24,
) -> Agent:
    """
    Create the Agent that chooses the next configuration experiment.

    :param chat_generator: Generator used to reason about experiment results and propose configuration changes.
    :param docs_toolset: Optional Haystack documentation tools available to the Agent.
    :param system_prompt: Optional replacement for the default optimizer instructions. What this environment has
        installed is appended either way, since it constrains every configuration the optimizer can propose.
    :param additional_instructions: Guidance appended to the instructions, for what a good configuration looks like
        in one specific harness. The instructions themselves stay free of any assumption about what the reference
        configuration does, so domain knowledge belongs here rather than in a rewritten replacement.
    :param max_agent_steps: Maximum number of Agent steps used to produce one proposal.
    :returns: The configured optimizer Agent.
    """
    instructions = system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT
    # Headed like the sections above them, so an appended block reads as its own section rather than as a
    # continuation of whatever the instructions happened to end on.
    instructions = f"{instructions}\n\n## This environment\n\n{describe_environment()}"
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n## This harness\n\n{additional_instructions.strip()}"
    # Measured on the retrieval harness over twenty labelled cases: this model reached a better configuration than
    # one costing ten times as much, and a whole experiment on it costs less than a single one of the other's
    # turns. The optimizer's turns are most of what an experiment costs once its cases are cheap, so what it is
    # worth paying for them is a question the harness can answer rather than assume.
    generator = chat_generator or OpenAIResponsesChatGenerator(
        model="gpt-5.6-luna",
        timeout=180.0,
        max_retries=5,
        generation_kwargs={"prompt_cache_key": OPTIMIZER_PROMPT_CACHE_KEY, "reasoning": {"effort": "low"}},
    )
    return Agent(
        chat_generator=generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
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


def _tool_specifications(reference: Optimizable) -> list[dict[str, Any]]:
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
    for key in ("mean_recall", "mean_precision", "mean_retrieved", "mean_queries"):
        if key in details:
            parts.append(f"{key.removeprefix('mean_')} {details[key]:.2f}")
    if (summary := details.get(CASE_SUMMARY_KEY)) is not None:
        parts.append(f"{summary.get('passed')}/{summary.get('cases')} cases clean")
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
        lines.append(_headline(entry.get("metrics")))
        if failure := entry.get("failure"):
            lines.append(f"failed to evaluate: {failure}")
        if rationale := entry.get("rationale"):
            lines.append(f"hypothesis: {rationale}")
        if diff := entry.get("diff"):
            lines.append(_fenced(diff.rstrip(), "diff"))
        entries.append("\n\n".join(lines))
    return "\n\n".join(entries)


def propose_candidate(
    optimizer_agent: Agent,
    workspace: ConfigurationWorkspace,
    reference: Optimizable,
    reference_runs: list[AgentRunRecord],
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
    # Three messages in order of how often they change, so the stable ones stay a reusable cached prefix: what is
    # being optimized and how it is judged never changes, the outcomes only ever gain an entry, and everything
    # that differs every turn is last.
    context = "".join(
        [
            _section("Objectives", json.dumps(objectives.to_dict())),
            _section("Known model prices", json.dumps(pricing.to_dict())),
            _section("Tools available to the reference", json.dumps(_tool_specifications(reference=reference))),
            _section(
                "How the reference behaved",
                "\n\n".join(
                    _fenced(json.dumps(digest_agent_run(result=record.outputs, policy=digest_policy), default=str))
                    for record in reference_runs[:3]
                    if record.outputs
                ),
            ),
            # Described case by case only while it is the only thing measured; once a candidate exists, the most
            # recent one is the configuration worth reading in that much detail.
            _section(
                "Reference measurement",
                json.dumps(baseline.to_dict() if not history else summarize_case_details(payload=baseline.to_dict())),
            ),
        ]
    )
    outcomes = _section("Outcomes so far", _render_outcomes(history=summarize_case_details(payload=history)))
    recent = history[-history_digest_window:] if history_digest_window > 0 else []
    current = "".join(
        [
            _section(
                "Budget",
                "unknown"
                if remaining_evaluations is None
                else f"{remaining_evaluations} evaluations remain, including this one.",
            ),
            _section("Most recent outcome in full", "\n\n".join(json.dumps(entry, default=str) for entry in recent)),
            _section(
                "candidate.yaml",
                f"revision `{configuration['revision']}`, edited from `{configuration['parent_id'][:12]}`\n\n"
                # Reproduced as a fenced block rather than a JSON string: this is the text the editing tools match
                # against, and a JSON string would show it with its newlines and quotes escaped.
                + _fenced(configuration["yaml"], "yaml"),
            ),
        ]
    )
    # Avoid serializing file-bound callables or cloning generator clients.
    agent = Agent(
        chat_generator=optimizer_agent.chat_generator,
        tools=[*optimizer_agent.tools, *workspace.tools()],
        system_prompt=optimizer_agent.system_prompt,
        max_agent_steps=optimizer_agent.max_agent_steps,
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
