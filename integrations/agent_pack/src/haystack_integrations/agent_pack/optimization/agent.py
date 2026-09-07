# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The Agent that chooses optimization experiments and the requests made to it."""

import json
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
from haystack.utils import _serialize_value_with_schema

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
    RunDigestPolicy,
    digest_agent_run,
    strip_run_digests,
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
Optimize a Haystack Agent through measured experiments. The configuration is in candidate.yaml, a one-component
Pipeline containing the Agent named 'agent'. Use read_config and edit_config to edit YAML directly. You may change
prompts, models, parameters, hooks, and add, remove or replace entire tools and pipelines. Each edit requires the
latest revision and a unique exact text match. The tools can only edit that file.

Use validate_config and repair errors before submit_candidate. Validation constructs the Agent but does not run
or warm it up. Submit one hypothesis per turn with a rationale.

Spend the evaluations. `remaining_evaluations` is a budget, not a limit to stay under, and an unused one is a
measurement nobody will ever take. A disappointing result is a finding about one hypothesis and says nothing about
whether others are left; the run that has just regressed is usually the one with the most still to learn. When the
obvious parameters have been tried, the configuration is still open: a component's prompt, what a component is
asked to produce rather than how much, the shape of the pipeline, and components not yet in it. Reach for finish
only when you can say what you considered and why none of it is worth measuring — it takes that reason as an
argument, and ending the search is the one decision the experiment cannot revisit.
Plain text does not submit a candidate. Invalid drafts and duplicates do not spend evaluation slots, but editing
steps are bounded. Edits continue from the last submitted candidate. Use restore_candidate with a history ID or
'reference' to start from a different base. Once a candidate passes the gates, vary one thing at a time against it.
Combine changes when they need to move together, and combine the change you are measuring with cleanups that cannot
plausibly interact with it: `remaining_evaluations` counts submissions and each one costs a full pass over the
evaluation set. Removing an unused tool also removes its schema from model input.

A component's default prompt is part of the configuration and is one of the most productive things to change. It
was written for that component's general case, not for what is being measured here, and a default that quietly
mismatches the task costs quality without ever failing: read the prompt in the YAML before assuming it fits.

Read a limit against what the run actually did with it. A component producing less than its own limit allows is
leaving that room unspent, and the reason is usually in its prompt rather than in the number. A limit reached on
every case is the opposite: it is binding, and what it truncates is invisible until it is raised.

Use inspect_component and optional documentation tools to learn installed components and their serialization.
A ComponentTool can become a PipelineTool: connect retriever.documents to ranker.documents, map query to both query
inputs, filters to the retriever, and ranker.documents to the tool documents output. Preserve outputs_to_state and
formatting handlers required by the harness. Do not invent serialization shapes or assume a package is installed.

Quality is a hard gate. Known prices are informational rather than an allowlist. Unpriced or incomplete usage
cannot win a cost optimization. Read gate failures and run evidence before choosing the next experiment.
Run evidence is compressed: truncated results and incomplete listings are explicitly marked.
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
        Agent does, so domain knowledge belongs here rather than in a rewritten replacement.
    :param max_agent_steps: Maximum number of Agent steps used to produce one proposal.
    :returns: The configured optimizer Agent.
    """
    instructions = system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT
    instructions = f"{instructions}\n\n{describe_environment()}"
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n{additional_instructions.strip()}"
    generator = chat_generator or OpenAIResponsesChatGenerator(
        model="gpt-5.6-terra",
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
    :returns: Submitted snapshot, or None after finish or exhaustion of the proposal step budget.
    """
    workspace.begin_turn()
    request = {
        "known_model_prices": pricing.to_dict(),
        "objectives": objectives.to_dict(),
        "remaining_evaluations": remaining_evaluations,
        # Exactly one configuration is described case by case: the most recently measured one. On the first turn
        # that is the reference, because nothing else has been measured yet. Afterwards it is the last candidate,
        # which `recent_outcomes_in_detail` carries, and re-sending the reference's listing every turn would spend
        # the budget describing a configuration that has since been superseded. How the reference behaved is not
        # lost with it: `reference_runs` carries its recorded runs separately.
        "baseline": baseline.to_dict() if not history else summarize_case_details(payload=baseline.to_dict()),
        "available_tools": _tool_specifications(reference=reference),
        "reference_runs": [
            {
                "inputs": _serialize_value_with_schema(payload=record.inputs)["serialized_data"],
                "outputs": digest_agent_run(result=record.outputs, policy=digest_policy),
            }
            for record in reference_runs[:3]
        ],
    }
    context_text = json.dumps(request, default=str)
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
            ChatMessage.from_user(text=context_text),
            ChatMessage.from_user(
                text=json.dumps(
                    {"outcomes": summarize_case_details(payload=strip_run_digests(payload=history))}, default=str
                )
            ),
            ChatMessage.from_user(
                text=json.dumps(
                    {
                        "recent_outcomes_in_detail": history[-history_digest_window:]
                        if history_digest_window > 0
                        else [],
                        "workspace": workspace.read_config(),
                    },
                    default=str,
                )
            ),
        ]
    )
    # What the turn spent, and on what. The journal records the diff a turn produced but nothing about how it got
    # there, so without this there is no way to tell an optimizer that considered a change and rejected it from one
    # that never looked. `exit_reason` matters on its own: a turn that ends on max_agent_steps without submitting
    # stops the whole experiment, not just itself.
    logger.info(
        "optimizer turn: steps={steps}/{budget} exit={exit_reason} calls={calls} usage={usage}",
        steps=result.get("step_count"),
        budget=agent.max_agent_steps,
        exit_reason=result.get("exit_reason"),
        calls=[call.tool_name for message in result.get("messages") or [] for call in message.tool_calls],
        usage=result.get("token_usage"),
    )
    return workspace.submitted
