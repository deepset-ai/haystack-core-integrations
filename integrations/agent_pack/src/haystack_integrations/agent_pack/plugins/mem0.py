# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from haystack.lazy_imports import LazyImport

from haystack_integrations.agent_pack.plugins.plugin import AgentPlugin

if TYPE_CHECKING:
    from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore

with LazyImport(message="Run 'pip install mem0-haystack' to use Mem0MemoryPlugin.") as mem0_import:
    from haystack_integrations.tools.mem0 import Mem0MemoryRetrieverTool, Mem0MemoryWriterTool

_DEFAULT_INPUTS_FROM_STATE = {"user_id": "user_id"}

_MEMORY_INSTRUCTIONS = """Use the long-term memory tools deliberately:
- At the beginning of a turn, call `retrieve_memories` when stored user context or preferences could be relevant.
- Store only new or changed durable, user-specific facts and preferences with `store_memory`; do not store transient
  requests, duplicate facts, retrieved tool results, or assistant-generated conclusions.
- Treat retrieved memories as potentially useful context, not authoritative factual evidence. Continue to obey the
  Agent's primary evidence and citation requirements, including any document-only or source-only restrictions."""


class Mem0MemoryPlugin(AgentPlugin):
    """Add Mem0-backed long-term memory tools and their required Agent State fields."""

    def __init__(
        self,
        *,
        memory_store: Mem0MemoryStore,
        top_k: int = 5,
        inputs_from_state: dict[str, str] | None = None,
    ) -> None:
        """
        Create the Mem0 memory plugin.

        :param memory_store: Mem0 store shared by the retrieval and writer tools.
        :param top_k: Default maximum number of memories returned by a focused search.
        :param inputs_from_state: Mapping from Agent State keys to Mem0 tool parameters. Every state key is registered
            as a string. Defaults to `{"user_id": "user_id"}`.
        """
        mem0_import.check()
        state_mapping = dict(_DEFAULT_INPUTS_FROM_STATE if inputs_from_state is None else inputs_from_state)
        super().__init__(
            name="mem0_memory",
            tools=[
                Mem0MemoryRetrieverTool(memory_store=memory_store, top_k=top_k, inputs_from_state=state_mapping),
                Mem0MemoryWriterTool(memory_store=memory_store, inputs_from_state=state_mapping),
            ],
            state_schema={state_key: {"type": str} for state_key in state_mapping},
            prompt_instructions=_MEMORY_INSTRUCTIONS,
        )
