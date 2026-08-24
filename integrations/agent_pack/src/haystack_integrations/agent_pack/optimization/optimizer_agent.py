# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for a skill-guided harness optimizer Agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from haystack.components.agents import Agent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.lazy_imports import LazyImport
from haystack.skill_stores.file_system import FileSystemSkillStore
from haystack.tools import SkillToolset, Toolset

if TYPE_CHECKING:
    from haystack_integrations.tools.mcp import MCPToolset

with LazyImport(message="Install 'mcp-haystack' to use the optional Haystack documentation toolset.") as mcp_import:
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

HAYSTACK_DOCS_MCP_URL = "https://docs.haystack.deepset.ai/api/mcp"

_OPTIMIZER_SYSTEM_PROMPT = """
You optimize Haystack Agent harnesses. Always load the haystack-agent-building skill before proposing a candidate.
Use the Haystack docs tool when current API details are uncertain and it is available.

You may only propose the typed recipe kinds documented by the skill. Never return Python code, import paths,
arbitrary serialized components, credentials, or deployment instructions. Return a JSON array of recipe objects and
nothing else. Prefer the smallest change that can meet the supplied quality, sovereignty, cost, and latency goals.
""".strip()


def bundled_agent_building_skills_path() -> Path:
    """Return the filesystem location of Agent Pack's bundled optimizer skills."""
    return Path(__file__).parent / "skills"


def create_haystack_docs_toolset(*, eager_connect: bool = False) -> MCPToolset:
    """
    Create an optional MCP toolset for the public Haystack documentation server.

    Requires the optional ``mcp-haystack`` package. The public server exposes ``search_haystack_docs`` and does not
    require credentials.
    """
    mcp_import.check()
    return MCPToolset(
        server_info=StreamableHttpServerInfo(url=HAYSTACK_DOCS_MCP_URL),
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
    """Create an optimizer Agent with a bundled Haystack-building skill and optional live documentation access."""
    skills = SkillToolset(FileSystemSkillStore(bundled_agent_building_skills_path()))
    tools: list[Toolset] = [skills]
    if docs_toolset is not None:
        tools.append(docs_toolset)
    return Agent(
        chat_generator=chat_generator,
        tools=tools,
        system_prompt=system_prompt or _OPTIMIZER_SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
    )
