# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for the harness optimizer Agent."""

from typing import TYPE_CHECKING

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset

from haystack_integrations.agent_pack.optimization.prompts import HARNESS_OPTIMIZER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

_DEFAULT_TIMEOUT = 180.0
_DEFAULT_MAX_RETRIES = 5

with LazyImport(message="Install 'mcp-haystack' to use the Haystack documentation MCP server.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo


def _default_llm(model: str) -> OpenAIResponsesChatGenerator:
    """
    A default OpenAI Responses-API generator with the pack's timeout/retry settings.

    Reasoning effort is left unset, so it takes the provider's default rather than the "low" the Advanced RAG agent
    asks for. Proposing a change is one call that decides what a whole experiment spends its budget measuring, so
    the quality of the judgement is worth more here than the latency of a single reply.

    :param model: The OpenAI model name.
    :returns: The generator.
    """
    return OpenAIResponsesChatGenerator(model=model, timeout=_DEFAULT_TIMEOUT, max_retries=_DEFAULT_MAX_RETRIES)


def create_haystack_documentation_mcp_toolset(*, eager_connect: bool = False) -> "MCPToolset":
    """
    Create a toolset backed by the public Haystack documentation MCP server.

    Requires the optional `mcp-haystack` package. The Haystack documentation MCP server exposes
    `search_haystack_docs` and needs no credentials.

    :param eager_connect: Whether to connect to the Haystack documentation MCP server when the toolset is created
        rather than on first use.
    :returns: A toolset exposing only the documentation search tool.
    """
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url="https://docs.haystack.deepset.ai/api/mcp"),
        tool_names=["search_haystack_docs"],
        eager_connect=eager_connect,
    )


def create_harness_optimizer_agent(
    *,
    chat_generator: ChatGenerator | None = None,
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 12,
) -> Agent:
    """
    Create the Agent that proposes changes to another Agent.

    Two Agents are involved in an optimization experiment and they are easy to confuse: the one being optimized,
    which the experiment measures and rebuilds variants of, and this one, which only reads a description of it and
    answers with the changes worth trying. This one never runs the harness and is never itself optimized.

    The recipe language and the rules for choosing between transformations are baked into the system prompt, so this
    Agent needs no tools to know them. Pass `docs_toolset` to additionally let it look up current Haystack API
    details while it reasons; `create_haystack_documentation_mcp_toolset` provides that from the Haystack
    documentation MCP server.

    Structured output is configured by `HarnessOptimizerAgentProposer` per request, because the response schema is
    generated from the approved asset catalog rather than fixed: pass its `structured_output_key` to enable it. The
    proposer validates every reply against that same schema regardless.

    :param chat_generator: The generator this Agent reasons with. Defaults to `OpenAIResponsesChatGenerator` on
        `gpt-5.4`. Structured output is configured per request by the proposer, so it does not need to be set here.
    :param docs_toolset: Optional read-only documentation toolset, for example the Haystack documentation MCP
        server from `create_haystack_documentation_mcp_toolset`.
    :param system_prompt: Replacement system prompt. `HARNESS_OPTIMIZER_SYSTEM_PROMPT` is used when omitted.
    :param max_agent_steps: Step budget for one proposal request.
    :returns: The optimizer Agent.
    """
    return Agent(
        chat_generator=chat_generator or _default_llm("gpt-5.4"),
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )
