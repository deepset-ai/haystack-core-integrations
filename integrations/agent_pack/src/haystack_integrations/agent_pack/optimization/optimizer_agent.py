# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for a skill-guided harness optimizer Agent."""

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

_OPTIMIZER_SYSTEM_PROMPT = """
You optimize Haystack Agent harnesses. Always load the haystack-agent-building skill before proposing a candidate.
Use the Haystack docs tool when current API details are uncertain and it is available.

You may only propose the typed recipe kinds documented by the skill, using only the models listed in
`approved_models` and the tools listed in `approved_tools`. Never return Python code, import paths,
arbitrary serialized components,
credentials, or deployment instructions. Return a JSON array of recipe objects and nothing else: no prose, no code
fences. Prefer the smallest change that can meet the supplied quality, cost, and latency goals.
""".strip()


def bundled_agent_building_skills_path() -> Path:
    """
    Return the filesystem location of Agent Pack's bundled optimizer skills.

    :returns: The directory holding the skills shipped inside the package.
    """
    return Path(__file__).parent / "skills"


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
    Create an optimizer Agent with a bundled Haystack-building skill and optional live documentation access.

    The Agent is expected to answer with a JSON array of typed recipes, so configure structured output on the
    generator you pass in. `HarnessOptimizerAgentProposer` recovers the array from a free-text reply as a fallback
    and validates every proposal through `recipe_from_dict` regardless, but constraining the response removes the
    prose and code fences that recovery has to work around.

    `RECIPE_PROPOSAL_JSON_SCHEMA` describes the expected shape. It is not marked strict, because the per-kind field
    sets form a union a strict schema cannot express cleanly; it pins the response to an object holding an array of
    recipe objects with a known `kind`, and `recipe_from_dict` remains the authoritative validator.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator

    from haystack_integrations.agent_pack.optimization import create_harness_optimizer_agent
    from haystack_integrations.agent_pack.optimization.recipes import RECIPE_PROPOSAL_JSON_SCHEMA

    optimizer = create_harness_optimizer_agent(
        chat_generator=OpenAIResponsesChatGenerator(
            model="gpt-5",
            generation_kwargs={
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "harness_optimizer_proposal",
                        "schema": RECIPE_PROPOSAL_JSON_SCHEMA,
                        "strict": False,
                    }
                }
            },
        )
    )
    ```

    `OpenAIChatGenerator` takes the same schema under a different key, as
    `generation_kwargs={"response_format": {"type": "json_schema", "json_schema": {...}}}`. Either generator also
    accepts a Pydantic model instead, through `text_format` and `response_format` respectively.

    Structured output does not constrain tool calls, only the final text reply — which is the reply the proposal is
    read from, since the Agent exits on text.

    :param chat_generator: The generator the optimizer reasons with. Configure structured output on it as above.
    :param docs_toolset: Optional read-only documentation toolset, for example from `create_haystack_docs_toolset`.
    :param system_prompt: Replacement system prompt. The bundled instructions are used when omitted.
    :param max_agent_steps: Step budget for one proposal request.
    :returns: The optimizer Agent.
    """
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
