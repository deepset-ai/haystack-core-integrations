# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The Agent that chooses optimization experiments and the requests made to it."""

import json
from typing import TYPE_CHECKING, Any

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
far. Run evidence is a digest, so a tool result may be a prefix: a truncated result and an incomplete listing both
say so, and a listing that reports omitted content is never exhaustive. Choose the most informative next
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
    :param system_prompt: Optional replacement for the default optimizer instructions.
    :param additional_instructions: Guidance appended to the instructions, for what a good configuration looks like
        in one specific harness. The instructions themselves stay free of any assumption about what the reference
        Agent does, so domain knowledge belongs here rather than in a rewritten replacement.
    :param max_agent_steps: Maximum number of Agent steps used to produce one proposal.
    :returns: The configured optimizer Agent.
    """
    instructions = system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n{additional_instructions.strip()}"
    generator = chat_generator or OpenAIResponsesChatGenerator(
        model="gpt-5.6-sol",
        timeout=180.0,
        max_retries=5,
        # Requests carrying the same key are routed together, which is what makes a reusable prefix likely to be
        # found in cache. It identifies this prompt family and its shape, so it changes when the request layout
        # does. Provider-specific, hence only on the generator this function owns.
        generation_kwargs={"prompt_cache_key": OPTIMIZER_PROMPT_CACHE_KEY},
    )
    return Agent(
        chat_generator=generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=instructions,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
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


def _log_cache_reuse(message: ChatMessage) -> None:
    """
    Report how much of the request the provider served from cache.

    The reusable prefix is the point of the request's layout, so whether it is actually being reused should be
    observable rather than assumed.

    :param message: The optimizer's reply, whose metadata carries provider usage.
    """
    usage = (message.meta or {}).get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    if (cached := details.get("cached_tokens")) is None or not (total := usage.get("input_tokens")):
        return
    logger.info(
        "Optimizer request reused {cached} of {total} input tokens from cache ({share:.0%}).",
        cached=cached,
        total=total,
        share=cached / total,
    )


def _bounded_history(history: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """
    Keep tool traces for the most recent outcomes and measurements for all of them.

    :param history: Every outcome observed so far, oldest first.
    :param window: How many of the most recent outcomes keep their traces.
    :returns: The history with older traces removed.
    """
    if window <= 0:
        return [strip_run_digests(payload=entry) for entry in history]
    keep = len(history) - window
    return [strip_run_digests(payload=entry) if index < keep else entry for index, entry in enumerate(history)]


def propose_mutation(
    optimizer_agent: Agent,
    reference: Agent,
    reference_runs: list[AgentRunRecord],
    pricing: ModelPriceCatalog,
    objectives: OptimizationObjectives,
    baseline: EvaluationMetrics,
    history: list[dict[str, Any]],
    digest_policy: RunDigestPolicy | None = None,
    history_digest_window: int = 2,
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
    :param history_digest_window: How many of the most recent outcomes keep their tool traces. A history is
        cumulative, so keeping every trace would grow the request with every turn; the measurements themselves are
        kept for all of them.
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
        # Last on purpose. Everything above is identical on every turn of an experiment, so keeping the one growing
        # section at the end leaves that stable text as a reusable prompt prefix instead of shifting it each turn.
        "history": _bounded_history(history=history, window=history_digest_window),
    }
    experiment_history = request.pop("history")
    result = optimizer_agent.run(
        # The unchanging context and the growing history are sent as separate messages. Cache reuse needs the
        # rendered prefix to match, and a provider that marks cache breakpoints does so between messages, so the
        # boundary between what is stable and what grows has to be a message boundary rather than a key in one blob.
        messages=[
            ChatMessage.from_user(text=json.dumps(request, default=str)),
            ChatMessage.from_user(text=json.dumps({"history": experiment_history}, default=str)),
        ],
        generation_kwargs={"text_format": OptimizerDecision},
    )
    _log_cache_reuse(message=result["last_message"])
    text = result["last_message"].text
    if text is None:
        msg = "The harness optimizer Agent returned no structured decision text."
        raise ValueError(msg)
    return OptimizerDecision.model_validate_json(text).mutation
