# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
An MCP server that exposes the Haystack deep research agent as a single MCP tool.

This wraps the *real* agent from `haystack_integrations.agent_pack.create_deep_research_agent`
and serves it over the Model Context Protocol, so any MCP client (Claude Code, Claude Desktop,
Codex, ...) can call it. The client does not re-implement the agent; it calls the `deep_research`
tool with a question and gets back a cited markdown report.

Run it directly for local development:

    OPENAI_API_KEY=... TAVILY_API_KEY=... python -m haystack_integrations.agent_pack.mcp.deep_research

or, once installed, via the `agent-pack-deep-research-mcp` console script (see this package's
README for zero-install `uvx` invocation and client config).
"""

import os

import anyio
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from mcp.server.fastmcp import Context, FastMCP

from haystack_integrations.agent_pack import create_deep_research_agent

mcp = FastMCP("agent-pack-deep-research")

# Build the agent once and reuse it across tool calls. Construction is cheap (no network), but
# reusing a single warm instance is the main advantage an MCP server has over the one-shot
# script: the process stays up, so repeated queries in a session skip re-construction entirely.
# Keyed by `max_subtopics` so a caller that overrides breadth gets its own cached instance.
_agents: dict[int | None, Agent] = {}


def _get_agent(max_subtopics: int | None) -> Agent:
    if max_subtopics not in _agents:
        _agents[max_subtopics] = (
            create_deep_research_agent()
            if max_subtopics is None
            else create_deep_research_agent(max_subtopics=max_subtopics)
        )
    return _agents[max_subtopics]


@mcp.tool()
async def deep_research(question: str, max_subtopics: int | None = None, ctx: Context | None = None) -> str:
    """
    Research a question on the web and return a structured, cited markdown report.

    The agent splits the question into sub-topics, delegates each to an isolated sub-researcher
    that searches the web and reads pages, then writes a report with inline `[text](url)`
    citations. This is a multi-minute, multi-agent job — use it for thorough investigations, not
    quick factual lookups.

    :param question: The research question. Be specific about scope, timeframe, and angle.
    :param max_subtopics: Optional cap on how many sub-questions to investigate (breadth).
    :returns: The final report as markdown, with inline citations.
    """
    missing = [key for key in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)]
    if missing:
        # Raised as an MCP tool error the client surfaces to the user/model.
        msg = f"{' and '.join(missing)} not set in the server environment"
        raise ValueError(msg)

    if ctx is not None:
        await ctx.info(f"Starting deep research: {question!r}")

    agent = _get_agent(max_subtopics)
    # agent.run is synchronous and long-running; run it in a worker thread so the server's event
    # loop stays responsive (able to report progress, handle cancellation).
    result = await anyio.to_thread.run_sync(lambda: agent.run(messages=[ChatMessage.from_user(question)]))

    if ctx is not None:
        await ctx.info(f"Research complete ({len(result.get('notes', []))} sub-topics investigated)")

    return result["report"]


def main() -> None:
    """Console-script entry point. Serves the tool over stdio (the default MCP transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
