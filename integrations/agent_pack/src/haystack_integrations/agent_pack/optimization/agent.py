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

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, content_digest
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import AgentMutation, OptimizerDecision
from haystack_integrations.agent_pack.run_digest import RunDigestPolicy, digest_agent_run, strip_run_digests

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

logger = logging.getLogger(__name__)

# Version the routing key with the request layout: a changed layout is a different reusable prefix.
OPTIMIZER_PROMPT_CACHE_KEY = "haystack-harness-optimizer-v1"

HARNESS_OPTIMIZER_SYSTEM_PROMPT = """
You optimize a Haystack Agent configuration through a measured sequence of experiments. On every turn you receive
the complete serialized reference Agent configuration, the tools that Agent can call, a digest of successful
reference runs, known model prices, optimization objectives, a baseline measurement, and all candidate outcomes so
far. Outcomes arrive twice: once as a record of every one measured, and once as the most recent few repeated with
the tool traces of their runs. Run evidence is a digest, so a tool result may be a prefix: a truncated result and an
incomplete listing both say so, and a listing that reports omitted content is never exhaustive. Choose the most
informative next
configuration mutation based on that evidence. You may change any part of the serialized Agent configuration.
Return null when no worthwhile experiment remains.

Express edits as ordered RFC 6901 JSON Pointer operations. `set` writes a scalar. `create_object` and `create_array`
create containers that later operations can populate. `remove` deletes a value. `copy` deep-copies any existing
configuration subtree. Array path `-` appends. Escape `~` as `~0` and `/` as `~1` in path segments. Every mutation is
applied to the unchanged reference configuration, not to the preceding candidate. Every operation has `value` and
`from_path` fields: set unused fields to null; only `set` uses `value`, and only `copy` uses `from_path`.

The Agent's tools are an array of configurations like any other part of it. Editing one changes what that tool does
or how it describes itself, and removing one withdraws the tool entirely, so its description and argument schema
stop being sent to the model on every step. A tool the Agent does not need is therefore a cost as well as a choice,
and instructing the Agent in its prompt to avoid a tool leaves that cost in place.

`set_json` writes a whole subtree from JSON text in one operation, so a component can be replaced by a differently
shaped one rather than only retuned. A tool backed by a single component can become one backed by a retrieval
pipeline, for instance, by writing that pipeline's serialized form in place of the component's. Such a change
succeeds only if every type it names can be imported and constructed in this environment, so confirm the shape of
what you are writing with the documentation tools before spending a measurement on it; a configuration that cannot
be rebuilt is reported back to you as a failed candidate.

Quality is a hard gate. Optimize the requested primary measurement only among candidates likely to preserve quality.
Known prices are informational rather than an allowlist: you may select other models, but their measured cost cannot be
ranked until pricing is supplied. Use documentation tools before changing an unfamiliar component path or provider
generation argument. Learn from failed mutations and measurements, and do not repeat a resulting configuration.

Design every candidate so that its outcome is attributable. Combine changes only when they have to move together to
clear a gate, and once a candidate passes every gate, treat that candidate as the base and vary one thing at a time
against it. The measured failure names identify the cause, so read them before choosing what to change. A setting that
is cheaper or smaller per unit is not automatically cheaper per task, so measure such a change instead of assuming it.
""".strip()


def create_haystack_documentation_mcp_toolset(eager_connect: bool = False) -> "MCPToolset":
    """
    Create the optional read-only public Haystack documentation toolset.

    :param eager_connect: Connect to the documentation server immediately instead of on first use.
    :returns: A toolset exposing Haystack documentation search.
    """
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=["search_haystack_docs"],
        eager_connect=eager_connect,
    )


def create_harness_optimizer_agent(
    chat_generator: ChatGenerator | None = None,
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    additional_instructions: str | None = None,
    max_agent_steps: int = 12,
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
    # The mid-priced model rather than the top one: an optimizer turn reads a large assembled request, and input
    # tokens dominate what it costs, so the model choice here is worth about as much as everything the candidates
    # spend. Pass a generator to choose differently.
    generator = chat_generator or OpenAIResponsesChatGenerator(
        model="gpt-5.6-terra",
        timeout=180.0,
        max_retries=5,
        generation_kwargs={
            # Requests carrying the same key are routed together, which is what makes a reusable prefix likely to
            # be found in cache. It identifies this prompt family and its shape, so it changes when the request
            # layout does. Provider-specific, hence only on the generator this function owns.
            "prompt_cache_key": OPTIMIZER_PROMPT_CACHE_KEY,
            # Set rather than left to the provider's heavier default. One decision is made per turn, from evidence
            # already assembled and summarized, and reasoning tokens are billed as output on the most expensive
            # model in the experiment. `_log_optimizer_usage` reports what each turn actually spends, so raising
            # this is a measurable choice rather than a guess.
            "reasoning": {"effort": "low"},
        },
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


def _tool_specifications(reference: Agent) -> list[dict[str, Any]]:
    """
    Describe the tools the reference Agent can call.

    A serialized Agent does not reliably carry this: a `ComponentTool` serializes its parameter schema as null
    whenever the schema is derived from the wrapped component, and a `Toolset` that serializes a descriptor of
    itself carries no tool names at all. Without this, the only way to learn what a tool is called and what it
    accepts is to find one already invoked in a recorded run.

    :param reference: The Agent whose tools to describe.
    :returns: One `{name, description, parameters}` entry per tool, or an empty list when they cannot be read.
    """
    try:
        warm_up_tools(tools=reference.tools)
        return [tool.tool_spec for tool in flatten_tools_or_toolsets(tools=reference.tools)]
    except Exception as error:
        logger.warning("Could not describe the reference Agent's tools: {error}", error=error)
        return []


def _log_optimizer_usage(message: ChatMessage, prefix_digest: str) -> None:
    """
    Report what one optimizer turn spent.

    Choosing candidates is not free, and its cost appears nowhere in an experiment's measurements: the reported
    numbers describe the candidates, not the search that found them. Reporting input, cache reuse and reasoning
    tokens per turn is what makes the model and effort behind the optimizer a measurable choice.

    The digest of the unchanging message is reported alongside, because low reuse means two different things: an
    identical digest across turns points at the provider or at something ahead of the messages, while a digest
    that changes means the prefix was never stable to begin with.

    :param message: The optimizer's reply, whose metadata carries provider usage.
    :param prefix_digest: Digest of the message that is supposed to be identical on every turn.
    """
    usage = (message.meta or {}).get("usage") or {}
    if not (total := usage.get("input_tokens")):
        return
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    output = usage.get("output_tokens") or 0
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    logger.info(
        "optimizer turn: {total} input ({cached} cached, {share:.0%}), {output} output of which {reasoning} "
        "reasoning, prefix {prefix}",
        total=total,
        cached=cached,
        share=cached / total,
        output=output,
        reasoning=reasoning,
        prefix=prefix_digest,
    )


def _history_record(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Describe every outcome so far, without the tool traces.

    Each entry takes this form the turn it is created and is never revised, so the record only ever grows by
    appending. That is what a reusable prompt prefix requires: rewriting an earlier entry — which is what dropping
    its trace later amounts to — changes the text mid-message and ends the reuse from that point on.

    :param history: Every outcome observed so far, oldest first.
    :returns: The same outcomes with every tool trace removed.
    """
    return [strip_run_digests(payload=entry) for entry in history]


def _recent_outcomes(history: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """
    Repeat the most recent outcomes in full, tool traces included.

    Traces dominate an outcome's size and only the newest are worth it, so they travel separately from the record
    rather than being edited out of it later. These entries appear twice by design: once in the append-only record
    and once here with their detail. This is the only part of a request whose shape changes from turn to turn,
    which is why it is sent last.

    :param history: Every outcome observed so far, oldest first.
    :param window: How many of the most recent outcomes to repeat in full.
    :returns: The most recent outcomes, or nothing when none are wanted.
    """
    return list(history[-window:]) if window > 0 else []


def propose_mutation(
    optimizer_agent: Agent,
    reference: Agent,
    reference_runs: list[AgentRunRecord],
    pricing: ModelPriceCatalog,
    objectives: OptimizationObjectives,
    baseline: EvaluationMetrics,
    history: list[dict[str, Any]],
    digest_policy: RunDigestPolicy | None = None,
    history_digest_window: int = 1,
) -> AgentMutation | None:
    """
    Ask the optimizer Agent for the next structured configuration mutation.

    :param optimizer_agent: Agent that chooses the next configuration experiment.
    :param reference: Unchanged reference Agent and source configuration for every candidate.
    :param reference_runs: Reference inputs and outputs that candidates must preserve.
    :param pricing: Known model prices supplied as optimization context.
    :param objectives: Quality gates and primary optimization measurement.
    :param baseline: Measured reference Agent performance.
    :param history: Candidate mutations and outcomes observed so far.
    :param digest_policy: Caps applied when compressing the reference runs into tool-behaviour evidence.
    :param history_digest_window: How many of the most recent outcomes are repeated in full with their tool
        traces. Traces dominate an outcome's size, so keeping every one would grow the request with every turn;
        the measurements themselves are kept for every outcome. Each repeat costs its whole entry, aggregates
        included, and that repeat is the only part of a request that cannot be reused from turn to turn, so one is
        the default.
    :returns: The next mutation, or `None` when the optimizer chooses to stop.
    """
    request = {
        "reference_agent_configuration": reference.to_dict(),
        "known_model_prices": pricing.to_dict(),
        "objectives": objectives.to_dict(),
        "baseline": baseline.to_dict(),
        "available_tools": _tool_specifications(reference=reference),
        "successful_reference_runs": [
            {
                "inputs": _serialize_value_with_schema(payload=record.inputs)["serialized_data"],
                "outputs": digest_agent_run(result=record.outputs, policy=digest_policy),
            }
            for record in reference_runs[:3]
        ],
    }
    # The outcomes are deliberately not part of the message above: everything in it is identical on every turn of
    # an experiment, and keeping what grows out of it is what leaves its text reusable.
    context_text = json.dumps(request, default=str)
    record = {"outcomes": _history_record(history=history)}
    detail = {"recent_outcomes_in_detail": _recent_outcomes(history=history, window=history_digest_window)}
    # Three messages, ordered by how often each changes: context that never does, a record that only grows by
    # appending, then the newest outcomes in full. Cache reuse needs the rendered prefix to match, and a provider
    # that marks cache breakpoints does so between messages, so those boundaries have to be message boundaries.
    result = optimizer_agent.run(
        messages=[
            ChatMessage.from_user(text=context_text),
            ChatMessage.from_user(text=json.dumps(record, default=str)),
            ChatMessage.from_user(text=json.dumps(detail, default=str)),
        ],
        generation_kwargs={"text_format": OptimizerDecision},
    )
    _log_optimizer_usage(message=result["last_message"], prefix_digest=content_digest(payload=context_text))
    text = result["last_message"].text
    if text is None:
        msg = "The harness optimizer Agent returned no structured decision text."
        raise ValueError(msg)
    return OptimizerDecision.model_validate_json(text).mutation
