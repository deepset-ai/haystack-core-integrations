# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.everos import EverOSMemoryStore


@component
class EverOSMemoryRetriever:
    """Retrieve EverOS user or agent memory as system ChatMessages for a Haystack pipeline."""

    def __init__(
        self,
        *,
        memory_store: EverOSMemoryStore,
        top_k: int = 5,
        method: Literal["keyword", "vector", "hybrid", "agentic"] = "hybrid",
        include_profile: bool = False,
    ) -> None:
        """
        Initialize the retriever.

        :param memory_store: EverOS store to search.
        :param top_k: Default result limit per memory kind.
        :param method: EverOS retrieval method.
        :param include_profile: Include the user profile in user-memory searches.
        """
        self.memory_store = memory_store
        self.top_k = top_k
        self.method = method
        self.include_profile = include_profile

    @component.output_types(memories=list[ChatMessage])
    def run(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        radius: float | None = None,
        min_score: float | None = None,
        enable_llm_rerank: bool = False,
        include_unprocessed: bool = False,
    ) -> dict[str, list[ChatMessage]]:
        """
        Search EverOS memory.

        :param query: Retrieval query.
        :param user_id: User-memory owner. Mutually exclusive with `agent_id`.
        :param agent_id: Agent-memory owner. Mutually exclusive with `user_id`.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :param session_id: Optional session filter.
        :param filters: Optional Haystack metadata-filter expression.
        :param top_k: Runtime result-limit override.
        :param radius: Optional cosine-similarity threshold.
        :param min_score: Optional post-fusion score floor.
        :param enable_llm_rerank: Enable LLM reranking for compatible agent-memory searches.
        :param include_unprocessed: Include messages still waiting in the session buffer.
        :returns: Dictionary with `memories`, a list of system ChatMessages.
        """
        memories = self.memory_store.search_memories(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            project_id=project_id,
            session_id=session_id,
            filters=filters,
            method=self.method,
            top_k=top_k if top_k is not None else self.top_k,
            radius=radius,
            min_score=min_score,
            include_profile=self.include_profile,
            enable_llm_rerank=enable_llm_rerank,
            include_unprocessed=include_unprocessed,
        )
        return {"memories": memories}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
            top_k=self.top_k,
            method=self.method,
            include_profile=self.include_profile,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverOSMemoryRetriever":
        """Deserialize this component from a dictionary."""
        if data.get("init_parameters", {}).get("memory_store"):
            data["init_parameters"]["memory_store"] = EverOSMemoryStore.from_dict(
                data["init_parameters"]["memory_store"]
            )
        return default_from_dict(cls, data)
