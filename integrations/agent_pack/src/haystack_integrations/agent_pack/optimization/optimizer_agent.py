# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for the harness optimizer Agent."""

from typing import TYPE_CHECKING

from haystack.components.agents import Agent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.lazy_imports import LazyImport
from haystack.tools import Toolset

from haystack_integrations.agent_pack.optimization.prompts import HARNESS_OPTIMIZER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the optional Haystack documentation toolset.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo


def create_haystack_docs_toolset(*, eager_connect: bool = False) -> "MCPToolset":
    """
    Create an optional MCP toolset for the public Haystack documentation server.

    Requires the optional `mcp-haystack` package. The public server exposes `search_haystack_docs` and does not
    require credentials.

    :param eager_connect: Whether to connect when the toolset is created rather than on first use.
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
    chat_generator: ChatGenerator,
    docs_toolset: Toolset | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 12,
) -> Agent:
    """
    Create an optimizer Agent that proposes typed harness transformations.

    The recipe language and the rules for choosing between transformations are baked into the system prompt, so the
    Agent needs no tools to know them. Pass `docs_toolset` to additionally let it look up current Haystack API
    details while it reasons.

    Structured output is configured by `HarnessOptimizerAgentProposer` per request, because the response schema is
    generated from the approved asset catalog rather than fixed: pass its `structured_output_key` to enable it. The
    proposer validates every reply against that same schema regardless.

    :param chat_generator: The generator the optimizer reasons with. Configure structured output on it as above.
    :param docs_toolset: Optional read-only documentation toolset, for example from `create_haystack_docs_toolset`.
    :param system_prompt: Replacement system prompt. `HARNESS_OPTIMIZER_SYSTEM_PROMPT` is used when omitted.
    :param max_agent_steps: Step budget for one proposal request.
    :returns: The optimizer Agent.
    """
    return Agent(
        chat_generator=chat_generator,
        tools=[docs_toolset] if docs_toolset is not None else None,
        system_prompt=system_prompt or HARNESS_OPTIMIZER_SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )
