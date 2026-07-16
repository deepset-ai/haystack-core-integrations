# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.retrievers.types import TextRetriever
from haystack.document_stores.types import DocumentStore
from haystack.hooks import Hook, HookPoint
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool, Toolset, ToolsType

from haystack_integrations.agent_pack.advanced_rag import prompts
from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.advanced_rag.tools import (
    DocumentStoreToolset,
    make_retrieval_pipeline_tool,
    make_retriever_tool,
)

# The default system prompt renders today's date via the Jinja `{% now %}` tag, which needs `arrow`. Without this
# check, a missing `arrow` surfaces as a cryptic `TemplateSyntaxError: Encountered unknown tag 'now'` because
# `ChatPromptBuilder` silently skips the extension when `arrow` is not installed.
with LazyImport(message='Run "pip install arrow>=1.3.0"') as arrow_import:
    import arrow  # noqa: F401

_DEFAULT_TIMEOUT = 180.0
_DEFAULT_MAX_RETRIES = 5


def _default_llm(model: str) -> OpenAIResponsesChatGenerator:
    """
    A default OpenAI Responses-API generator with the pack's timeout/retry settings.

    Reasoning effort is set to "low": enough deliberate thinking for tool sequencing and filter construction, without
    compounding latency across the agent's steps (leaving it unset would default to the heavier "medium").

    :param model: The OpenAI model name.
    :returns: The generator.
    """
    return OpenAIResponsesChatGenerator(
        model=model,
        timeout=_DEFAULT_TIMEOUT,
        max_retries=_DEFAULT_MAX_RETRIES,
        generation_kwargs={"reasoning": {"effort": "low"}},
    )


def create_advanced_rag_agent(
    *,
    document_store: DocumentStore,
    retriever: TextRetriever | Pipeline | None = None,
    retrieval_pipeline_input_mapping: dict[str, list[str]] | None = None,
    retrieval_pipeline_output_mapping: dict[str, str] | None = None,
    llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 20,
    max_fetched_docs: int = 10,
    extra_tools: ToolsType | None = None,
    state_schema: dict[str, Any] | None = None,
    hooks: dict[HookPoint, list[Hook]] | None = None,
    raise_on_tool_invocation_failure: bool = False,
    tool_concurrency_limit: int = 4,
) -> Agent:
    """
    Create the advanced RAG agent.

    The agent answers questions from documents it retrieves out of the document store. Instead of guessing which
    metadata fields exist, it can inspect the store (fields, values, ranges) and construct a Haystack filter to
    narrow its retrieval when metadata helps — plain, unfiltered retrieval remains available when it doesn't. The
    answer cites the retrieved documents.

    The required `retriever` becomes the `search_documents` tool; `document_store` additionally feeds the three
    metadata inspection tools and must implement the metadata introspection methods (`get_metadata_fields_info`,
    `get_metadata_field_unique_values`, `get_metadata_field_min_max`).

    :param document_store: The document store the metadata inspection tools and the `fetch_documents_by_filter` tool
        run against.
    :param retriever: What retrieves for the `search_documents` tool (required). Either a standalone retriever
        component following the `TextRetriever` protocol, i.e. its `run` method accepts `query` and `filters`
        (e.g. `InMemoryBM25Retriever`, or an embedding retriever wrapped in `TextEmbeddingRetriever`), or a custom
        retrieval `Pipeline` (e.g. embedder -> retriever, or hybrid retrieval) — a pipeline additionally requires
        `retrieval_pipeline_input_mapping`. It should retrieve by relevance scoring (keyword or embedding-based) —
        direct, unscored fetching is already covered by the built-in `fetch_documents_by_filter` tool.
    :param retrieval_pipeline_input_mapping: Required when `retriever` is a `Pipeline`: maps the tool inputs to
        pipeline input sockets; must have exactly the keys "query" and "filters",
        e.g. `{"query": ["embedder.text"], "filters": ["retriever.filters"]}`.
    :param retrieval_pipeline_output_mapping: Optional when `retriever` is a `Pipeline`: maps pipeline output sockets
        to tool outputs, e.g. `{"retriever.documents": "documents"}`.
    :param llm: LLM that drives the agent loop. Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")` with low
        reasoning effort.
    :param system_prompt: Overrides the pre-made system prompt
        (`haystack_integrations.agent_pack.advanced_rag.prompts.SYSTEM_TEMPLATE`).
    :param max_agent_steps: Maximum steps for the agent loop. If the loop is cut off by this limit before writing an
        answer, an `after_run` hook (`BackupAnswerHook`) makes one extra LLM call to produce a best-effort answer from
        the evidence gathered so far, so `last_message` always carries a text answer.
    :param max_fetched_docs: Maximum number of documents `fetch_documents_by_filter` shows per fetch. A filter fetch is
        not bounded by a retriever's `top_k`, so this caps the tool result instead; the scored `search_documents` tool
        is bounded by the `top_k` configured on your retrieval components.
    :param extra_tools: Additional tools (or toolsets) for the agent, appended after the built-in document-store
        toolset and the retrieval tool.
    :param state_schema: Additional entries merged into the agent's state schema. The built-in `documents` entry
        (the accumulated retrieved documents) always takes precedence.
    :param hooks: Additional hooks per hook point, merged with the built-in hooks. For `after_run`, the built-in
        backup-answer hook runs first, so custom hooks see the final answer.
    :param raise_on_tool_invocation_failure: If True, a failing tool call raises instead of being returned to the LLM
        as an error message it can recover from (the default).
    :param tool_concurrency_limit: Maximum number of tool calls executed in parallel within one agent step.
    :returns: The advanced RAG `Agent`. Call it with the question as a user message,
        `agent.run(messages=[ChatMessage.from_user(question)])`; the answer is in `last_message` (a `ChatMessage`) and
        `documents` carries every document the agent retrieved during the run (deduplicated by id, in first-retrieved
        order) — the answer cites them by the first 8 characters of their id, e.g. `[doc a1b2c3d4]`. The standard Agent
        outputs `messages`, `step_count`, `token_usage` and `tool_call_counts` are also returned.
    """
    if system_prompt is None:
        # Only the default system prompt requires the `{% now %}` Jinja tag (and thus `arrow`).
        arrow_import.check()

    if retriever is None:
        msg = "`retriever` is required: pass a retriever component or a retrieval `Pipeline`."
        raise ValueError(msg)

    retrieval_tool: Tool
    if isinstance(retriever, Pipeline):
        if retrieval_pipeline_input_mapping is None:
            msg = "`retrieval_pipeline_input_mapping` is required when `retriever` is a `Pipeline`."
            raise ValueError(msg)
        retrieval_tool = make_retrieval_pipeline_tool(
            pipeline=retriever,
            input_mapping=retrieval_pipeline_input_mapping,
            output_mapping=retrieval_pipeline_output_mapping,
        )
    else:
        if retrieval_pipeline_input_mapping is not None or retrieval_pipeline_output_mapping is not None:
            msg = "The `retrieval_pipeline_*` arguments are only valid when `retriever` is a `Pipeline`."
            raise ValueError(msg)
        retrieval_tool = make_retriever_tool(retriever=retriever)

    llm = llm or _default_llm("gpt-5.4")

    tools: list[Tool | Toolset] = [
        DocumentStoreToolset(document_store, max_fetched_docs=max_fetched_docs),
        retrieval_tool,
    ]
    if extra_tools is not None:
        tools.extend([extra_tools] if isinstance(extra_tools, Toolset) else list(extra_tools))

    merged_hooks: dict[HookPoint, list[Hook]] = {"after_run": [BackupAnswerHook(generator=llm)]}
    for hook_point, point_hooks in (hooks or {}).items():
        merged_hooks.setdefault(hook_point, []).extend(point_hooks)

    return Agent(
        chat_generator=llm,
        system_prompt=system_prompt or prompts.SYSTEM_TEMPLATE,
        tools=tools,
        exit_conditions=["text"],
        max_agent_steps=max_agent_steps,
        state_schema={**(state_schema or {}), "documents": {"type": list[Document]}},
        hooks=merged_hooks,
        raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
        tool_concurrency_limit=tool_concurrency_limit,
    )
