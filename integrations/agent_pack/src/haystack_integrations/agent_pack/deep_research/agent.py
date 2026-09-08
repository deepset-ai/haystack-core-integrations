# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# The two agents and the composition that wires them into one deep research Agent.
#
# - researcher: a reusable sub-agent that researches ONE sub-question in its own isolated context and returns a
#   compressed, cited summary.
# - deep research agent:
#     - before starting the Agent loop, turns the user query into a research brief using the Scope Hook.
#     - delegates sub-questions to the researcher and accumulates the returned summaries in shared `State`.
#     - after the Agent loop, turns the brief plus collected notes into the final report using the Write Hook.

from haystack import logging
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import AgentTool, Tool

from haystack_integrations.agent_pack.deep_research import prompts
from haystack_integrations.agent_pack.deep_research.hooks import ScopeHook, WriteHook
from haystack_integrations.agent_pack.deep_research.tools import make_read_url_tool, think_tool

with LazyImport(message="Run 'pip install tavily-haystack'") as tavily_import:
    from haystack_integrations.tools.tavily import TavilyWebSearchTool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180.0
_DEFAULT_MAX_RETRIES = 5


def _default_llm(model: str) -> OpenAIResponsesChatGenerator:
    """A default OpenAI generator for a stage, with the pack's timeout/retry settings."""
    return OpenAIResponsesChatGenerator(model=model, timeout=_DEFAULT_TIMEOUT, max_retries=_DEFAULT_MAX_RETRIES)


def _collect_note(current: list[str] | None, message: ChatMessage) -> list[str]:
    """State handler: append a researcher's final summary (text) to `notes`."""
    current = current or []
    note = getattr(message, "text", None)  # message is the researcher's last ChatMessage
    if not note:
        logger.info("sub-researcher returned an empty note (likely hit its step cap); skipping")
        return current
    logger.info(
        "note collected (#{index}, {length} chars):\n{note}", index=len(current) + 1, length=len(note), note=note
    )
    return [*current, note]


def _make_researcher_agent(
    *,
    researcher_llm: ChatGenerator,
    page_summary_llm: ChatGenerator,
    search_tool: Tool | None,
    max_researcher_steps: int,
    max_page_chars: int,
) -> Agent:
    """
    The reusable sub-researcher.

    :param researcher_llm: LLM that drives the search/read/think loop and writes the summary.
    :param page_summary_llm: LLM used inside the `read_url` tool to summarize a page toward the question.
    :param search_tool: Web search tool. Defaults to `TavilyWebSearchTool(top_k=10)`.
    :param max_researcher_steps: Cap on the researcher's agent loop (search/read/think + final summary).
    :param max_page_chars: Raw page chars fed to the summarizer (pre-summarization cap).
    """
    if search_tool is None:
        tavily_import.check()
        search_tool = TavilyWebSearchTool(top_k=10)
    return Agent(
        chat_generator=researcher_llm,
        system_prompt=prompts.RESEARCHER_TEMPLATE,
        tools=[
            search_tool,
            make_read_url_tool(page_summary_llm=page_summary_llm, max_page_chars=max_page_chars),
            think_tool,
        ],
        exit_conditions=["text"],  # exits when it writes its summary
        max_agent_steps=max_researcher_steps,
    )


def create_deep_research_agent(
    *,
    llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 8,
    brief_llm: ChatGenerator | None = None,
    researcher_llm: ChatGenerator | None = None,
    search_tool: Tool | None = None,
    page_summary_llm: ChatGenerator | None = None,
    report_llm: ChatGenerator | None = None,
    max_researcher_steps: int = 20,
    max_concurrent_researchers: int = 5,
    max_subtopics: int = 5,
    max_page_chars: int = 50_000,
) -> Agent:
    """
    Create the deep research agent.

    To customize the returned agent further (e.g. add tools or hooks), use `Agent.clone`:
    `agent = agent.clone(tools=[*agent.tools, my_tool])`.

    :param llm: LLM that plans the investigation and delegates the sub-questions.
        Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
    :param system_prompt: Overrides the pre-made system prompt. `{{ max_subtopics }}` is replaced with the value of
        `max_subtopics`.
    :param max_agent_steps: Maximum steps for the agent loop (reflect -> delegate rounds).
    :param brief_llm: LLM that rewrites the user query into a focused research brief.
        Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
    :param researcher_llm: LLM that drives each sub-researcher's search/read/think loop.
        Defaults to `OpenAIResponsesChatGenerator("gpt-5.4-mini")`.
    :param search_tool: Web search tool used by each sub-researcher. Defaults to `TavilyWebSearchTool(top_k=10)`,
        which requires `tavily-haystack`. The pre-made researcher prompt refers to the tool as `web_search`.
    :param page_summary_llm: LLM used inside the `read_url` tool to summarize a fetched page toward
        the question. Defaults to `OpenAIResponsesChatGenerator("gpt-5.4-mini")`.
    :param report_llm: LLM that turns the brief plus collected notes into the final report.
        Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
    :param max_researcher_steps: Maximum steps for each sub-researcher's agent loop.
    :param max_concurrent_researchers: Maximum number of sub-researchers that run at the same time.
    :param max_subtopics: Maximum number of sub-questions the agent may delegate (breadth).
    :param max_page_chars: Maximum raw page characters fed to the summarizer, before summarization.
    :returns: The deep research `Agent`. Call it with the question as a user message,
        `agent.run(messages=[ChatMessage.from_user(question)])`; it returns a dict whose main output is
        `report` (the final markdown report, a `str`). The dict also carries the intermediate `brief`
        (`str`) and `notes` (`list[str]`), plus the standard Agent outputs `messages`, `last_message`,
        `step_count`, `token_usage` and `tool_call_counts`.
    """
    llm = llm or _default_llm("gpt-5.4")
    brief_llm = brief_llm or _default_llm("gpt-5.4")
    researcher_llm = researcher_llm or _default_llm("gpt-5.4-mini")
    page_summary_llm = page_summary_llm or _default_llm("gpt-5.4-mini")
    report_llm = report_llm or _default_llm("gpt-5.4")

    researcher = _make_researcher_agent(
        researcher_llm=researcher_llm,
        page_summary_llm=page_summary_llm,
        search_tool=search_tool,
        max_researcher_steps=max_researcher_steps,
        max_page_chars=max_page_chars,
    )

    research_subtopic = AgentTool(
        agent=researcher,
        name="research_subtopic",
        description=(
            "Research ONE focused sub-question in an isolated context. "
            "Pass the sub-question as a user message. "
            "Returns a compressed, cited summary of the findings."
        ),
        outputs_to_state={"notes": {"source": "last_message", "handler": _collect_note}},
    )

    scope = ScopeHook(
        generator=brief_llm,
        prompt_builder=ChatPromptBuilder(template=prompts.SCOPE_TEMPLATE, required_variables=["query"]),
    )
    write = WriteHook(
        generator=report_llm,
        prompt_builder=ChatPromptBuilder(template=prompts.WRITER_TEMPLATE, required_variables=["replies", "notes"]),
    )

    system_prompt = (system_prompt or prompts.ORCHESTRATOR_TEMPLATE).replace("{{ max_subtopics }}", str(max_subtopics))

    return Agent(
        chat_generator=llm,
        system_prompt=system_prompt,
        tools=[research_subtopic, think_tool],
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
        tool_concurrency_limit=max_concurrent_researchers,
        state_schema={"notes": {"type": list}, "brief": {"type": str}, "report": {"type": str}},
        hooks={"before_run": [scope], "after_run": [write]},
    )
