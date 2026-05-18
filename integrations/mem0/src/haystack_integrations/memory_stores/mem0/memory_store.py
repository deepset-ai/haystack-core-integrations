# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.memory_stores.mem0.errors import Mem0MemoryStoreError
from haystack_integrations.memory_stores.mem0.filters import _build_search_filters
from mem0 import MemoryClient

logger = logging.getLogger(__name__)


class Mem0MemoryStore:
    """
    A memory store backed by the Mem0 cloud API.

    Stores and retrieves ChatMessage-based memories scoped by user_id, run_id, agent_id, or app_id.
    The Mem0 client is created lazily on first use (or explicitly via warm_up()).
    Requires a Mem0 API key set via the MEM0_API_KEY environment variable or passed explicitly.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("MEM0_API_KEY"),
    ) -> None:
        """
        Initialize the Mem0 memory store.

        The Mem0 client is not created until warm_up() is called (or the first method that
        needs the client is invoked).

        :param api_key: The Mem0 API key. Defaults to the MEM0_API_KEY environment variable.
        """
        self.api_key = api_key
        self._client: MemoryClient | None = None

    def warm_up(self) -> None:
        """
        Create the Mem0 client. Called automatically on first use if not called explicitly.

        Calling this method explicitly is useful when you want to validate the API key
        or pre-connect before the first pipeline run.
        """
        if self._client is None:
            self._client = MemoryClient(api_key=self.api_key.resolve_value())

    @property
    def client(self) -> MemoryClient:
        """Return the initialized client, calling warm_up() if necessary."""
        self.warm_up()
        return self._client  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(self, api_key=self.api_key.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        if data.get("init_parameters"):
            deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        infer: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects to store as memories.
        :param user_id: User ID to scope these memories.
        :param run_id: Run ID to scope these memories.
        :param agent_id: Agent ID to scope these memories. Required for Mem0 to store assistant messages.
        :param app_id: App ID to scope these memories.
        :param infer: If True, Mem0 extracts memories from messages. If False, Mem0 stores message text as-is.
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client add method.
            Note: ChatMessage.meta is ignored because Mem0 doesn't support per-message metadata.
            Pass `metadata` as a kwarg to attach metadata to the whole batch instead.
        :returns: List of objects with `memory_id` and `memory` text for each stored memory.
        :raises Mem0MemoryStoreError: If the Mem0 API call fails.
        """
        ids = self._get_ids(user_id=user_id, run_id=run_id, agent_id=agent_id, app_id=app_id)
        mem0_messages = [{"content": msg.text, "role": msg.role.value} for msg in messages if msg.text]
        if not mem0_messages:
            logger.warning(
                "No valid messages to add after filtering out empty texts. Returning without calling Mem0 API."
            )
            return []

        added: list[dict[str, Any]] = []
        try:
            status = self.client.add(
                messages=mem0_messages,
                infer=infer,
                **ids,
                **kwargs,
            )
            if status and "results" in status:
                for result in status["results"]:
                    data = result.get("data")
                    memory_text = data.get("memory") if isinstance(data, dict) else result.get("memory")
                    added.append({"memory_id": result.get("id"), "memory": memory_text})
        except Exception as e:
            msg = f"Failed to add memories: {e}"
            raise Mem0MemoryStoreError(msg) from e
        return added

    def search_memories(
        self,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        Either `filters` or at least one of `user_id`, `run_id`, `agent_id`, or `app_id` must be provided.
        When both `filters` and IDs are provided, they are combined with an `AND` condition.

        :param query: Text query to search. If omitted, returns all memories matching the scope.
        :param filters: Haystack-style filters to apply. See https://docs.haystack.deepset.ai/docs/metadata-filtering
        :param top_k: Maximum number of results to return.
        :param user_id: User ID to scope the search.
        :param run_id: Run ID to scope the search.
        :param agent_id: Agent ID to scope the search.
        :param app_id: App ID to scope the search.
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client.
        :returns: List of ChatMessage (system role) objects containing the retrieved memories. User-provided
            Mem0 metadata is included in each message's meta. Mem0 retrieval fields such as `memory_id`, `user_id`,
            `score`, and timestamps are included under `meta["mem0"]`.
        :raises Mem0MemoryStoreError: If the Mem0 API call fails.
        """
        mem0_filters = _build_search_filters(
            filters=filters,
            user_id=user_id,
            run_id=run_id,
            agent_id=agent_id,
            app_id=app_id,
        )

        try:
            if not query:
                raw = self.client.get_all(filters=mem0_filters, **kwargs)
            else:
                raw = self.client.search(query=query, top_k=top_k, filters=mem0_filters, **kwargs)

            result_messages = []
            for memory in raw["results"]:
                meta = dict(memory["metadata"]) if memory["metadata"] else {}
                meta["mem0"] = {
                    "memory_id": memory.get("id"),
                    "user_id": memory.get("user_id"),
                    "agent_id": memory.get("agent_id"),
                    "app_id": memory.get("app_id"),
                    "run_id": memory.get("run_id"),
                    "score": memory.get("score"),
                    "score_breakdown": memory.get("score_breakdown"),
                    "categories": memory.get("categories"),
                    "created_at": memory.get("created_at"),
                    "updated_at": memory.get("updated_at"),
                }
                result_messages.append(ChatMessage.from_system(text=memory["memory"], meta=meta))
            return result_messages
        except Exception as e:
            msg = f"Failed to search memories: {e}"
            raise Mem0MemoryStoreError(msg) from e

    def _get_ids(
        self,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Return non-empty Mem0 entity IDs for scoped memory operations.

        Mem0 requires at least one entity ID when adding or searching memories without explicit filters.
        Valid entity IDs are `user_id`, `run_id`, `agent_id`, and `app_id`.

        :param user_id: User ID to include in the scope.
        :param run_id: Run ID to include in the scope.
        :param agent_id: Agent ID to include in the scope.
        :param app_id: App ID to include in the scope.
        :returns: Dictionary containing only provided entity IDs.
        :raises ValueError: If no entity ID is provided.
        """
        if not any([user_id, run_id, agent_id, app_id]):
            msg = "At least one of user_id, run_id, agent_id, or app_id must be provided."
            raise ValueError(msg)
        return {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id, "app_id": app_id}.items() if v
        }
