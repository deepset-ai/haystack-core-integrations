# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.components.retrievers.mem0.retriever import Mem0MemoryRetriever
from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore

_DEFAULT_RETRIEVER_DESCRIPTION = (
    "Search long-term memories relevant to a query. "
    "Use this tool whenever you need to recall information from past conversations or stored facts."
)

_DEFAULT_WRITER_DESCRIPTION = (
    "Store a piece of information as a long-term memory. "
    "Use this tool to persist important facts, preferences, or context for future conversations."
)


def create_mem0_memory_retriever_tool(
    store: Mem0MemoryStore,
    *,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    top_k: int = 5,
    name: str = "retrieve_memories",
    description: str = _DEFAULT_RETRIEVER_DESCRIPTION,
) -> Tool:
    """
    Create a Tool that searches Mem0 memories with pre-bound scoping IDs.

    The returned Tool exposes only `query` and `top_k` to the LLM — the
    `user_id`, `run_id`, and `agent_id` are bound at creation time so the
    model never needs to supply them.

    :param store: The Mem0MemoryStore instance to query.
    :param user_id: User ID to scope searches.
    :param run_id: Run ID to scope searches.
    :param agent_id: Agent ID to scope searches.
    :param top_k: Default number of memories to return.
    :param name: Tool name exposed to the LLM.
    :param description: Tool description exposed to the LLM.
    :returns: A Haystack Tool ready to be passed to an Agent.
    """
    retriever = Mem0MemoryRetriever(memory_store=store, top_k=top_k)

    def _retrieve(query: str, top_k: int = top_k) -> str:
        result = retriever.run(
            query,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        memories: list[ChatMessage] = result["memories"]
        if not memories:
            return "No memories found."
        return "\n".join(f"- {m.text}" for m in memories if m.text)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to find relevant memories."},
            "top_k": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": top_k,
            },
        },
        "required": ["query"],
    }

    return Tool(name=name, description=description, parameters=parameters, function=_retrieve)


def create_mem0_memory_writer_tool(
    store: Mem0MemoryStore,
    *,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    name: str = "store_memory",
    description: str = _DEFAULT_WRITER_DESCRIPTION,
) -> Tool:
    """
    Create a Tool that writes a memory to Mem0 with pre-bound scoping IDs.

    The returned Tool exposes only `text` to the LLM — the `user_id`,
    `run_id`, and `agent_id` are bound at creation time.

    :param store: The Mem0MemoryStore instance to write to.
    :param user_id: User ID to scope the stored memory.
    :param run_id: Run ID to scope the stored memory.
    :param agent_id: Agent ID to scope the stored memory.
    :param name: Tool name exposed to the LLM.
    :param description: Tool description exposed to the LLM.
    :returns: A Haystack Tool ready to be passed to an Agent.
    """

    def _store(text: str) -> str:
        result = store.add_memories(
            messages=[ChatMessage.from_user(text)],
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        count = len(result) if isinstance(result, list) else 0
        return f"Stored {count} memory item(s)."

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The information to store as a memory."},
        },
        "required": ["text"],
    }

    return Tool(name=name, description=description, parameters=parameters, function=_store)
