# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.chat_message import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install mem0ai'") as mem0_import:
    from mem0 import MemoryClient

logger = logging.getLogger(__name__)


class Mem0MemoryStore:
    """
    A memory store backed by the Mem0 cloud API.

    Stores and retrieves ChatMessage-based memories scoped by user_id, run_id, or agent_id.
    Requires a Mem0 API key (set via the MEM0_API_KEY environment variable or passed explicitly).
    """

    def __init__(self, *, api_key: Secret = Secret.from_env_var("MEM0_API_KEY")) -> None:
        """
        Initialize the Mem0 memory store.

        :param api_key: The Mem0 API key. Defaults to the MEM0_API_KEY environment variable.
        """
        mem0_import.check()
        self.api_key = api_key
        self.client = MemoryClient(api_key=self.api_key.resolve_value())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(self, api_key=self.api_key.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        infer: bool = True,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects to store as memories.
        :param infer: If True, Mem0 extracts facts asynchronously (may return pending status without IDs).
            If False, the full message is stored synchronously and IDs are returned immediately.
        :param user_id: User ID to scope these memories.
        :param run_id: Run ID to scope these memories.
        :param agent_id: Agent ID to scope these memories. Required for Mem0 to store assistant messages.
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client add method.
            Note: ChatMessage.meta is ignored because Mem0 doesn't support per-message metadata.
            Pass `metadata` as a kwarg to attach metadata to the whole batch instead.
        :returns: List of objects with `memory_id` and `memory` text for each stored memory.
        """
        ids = self._get_ids(user_id, run_id, agent_id)
        self.client.project.update(
            custom_instructions="Store all memories from the user and suggestions from the assistant."
        )
        mem0_messages = [{"content": msg.text, "role": msg.role.value} for msg in messages if msg.text]
        added: list[dict[str, Any]] = []
        try:
            status = self.client.add(messages=mem0_messages, infer=infer, **ids, **kwargs)
            if status and "results" in status:
                for result in status["results"]:
                    data = result.get("data")
                    memory_text = data.get("memory") if isinstance(data, dict) else result.get("memory")
                    added.append({"memory_id": result.get("id"), "memory": memory_text})
        except Exception as e:
            msg = f"Failed to add memories: {e}"
            raise RuntimeError(msg) from e
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
        include_memory_metadata: bool = False,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        Either `filters` or at least one of `user_id`, `run_id`, `agent_id` must be provided.
        When `filters` is given the ID parameters are ignored; Mem0 filters take precedence.

        :param query: Text query to search. If omitted, returns all memories matching the scope.
        :param filters: Haystack-style filters to apply. See https://docs.haystack.deepset.ai/docs/metadata-filtering
        :param top_k: Maximum number of results to return.
        :param user_id: User ID to scope the search.
        :param run_id: Run ID to scope the search.
        :param agent_id: Agent ID to scope the search.
        :param include_memory_metadata: If True, each returned ChatMessage's meta will include a
            `retrieved_memory_metadata` key containing the raw Mem0 memory object (memory_id, score, etc.).
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client.
        :returns: List of ChatMessage (system role) objects containing the retrieved memories.
        """
        if filters:
            mem0_filters = self.normalize_filters(filters)
        else:
            ids = self._get_ids(user_id, run_id, agent_id)
            mem0_filters = dict(ids) if len(ids) == 1 else {"AND": [{k: v} for k, v in ids.items()]}

        try:
            if not query:
                raw = self.client.get_all(filters=mem0_filters, **kwargs)
            else:
                raw = self.client.search(query=query, top_k=top_k, filters=mem0_filters, **kwargs)

            result_messages = []
            for memory in raw["results"]:
                meta = dict(memory["metadata"]) if memory["metadata"] else {}
                if include_memory_metadata:
                    meta["retrieved_memory_metadata"] = {k: v for k, v in memory.items() if k != "memory"}
                result_messages.append(ChatMessage.from_system(text=memory["memory"], meta=meta))
            return result_messages
        except Exception as e:
            msg = f"Failed to search memories: {e}"
            raise RuntimeError(msg) from e

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete all memories for the given scope.

        At least one of user_id, run_id, or agent_id must be provided.

        :param user_id: User ID whose memories to delete.
        :param run_id: Run ID whose memories to delete.
        :param agent_id: Agent ID whose memories to delete.
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client delete_all method.
        """
        ids = self._get_ids(user_id, run_id, agent_id)
        try:
            self.client.delete_all(**ids, **kwargs)
            logger.info("All memories deleted for scope {ids}", ids=ids)
        except Exception as e:
            msg = f"Failed to delete memories for scope {ids}: {e}"
            raise RuntimeError(msg) from e

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """
        Delete a single memory by ID.

        :param memory_id: The ID of the memory to delete.
        :param kwargs: Additional keyword arguments forwarded to the Mem0 client delete method.
        """
        try:
            self.client.delete(memory_id=memory_id, **kwargs)
            logger.info("Deleted memory {memory_id}", memory_id=memory_id)
        except Exception as e:
            msg = f"Failed to delete memory {memory_id}: {e}"
            raise RuntimeError(msg) from e

    def _get_ids(
        self,
        user_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        if not any([user_id, run_id, agent_id]):
            msg = "At least one of user_id, run_id, or agent_id must be provided."
            raise ValueError(msg)
        return {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v}

    @staticmethod
    def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Haystack-style filters to the Mem0 filter format.

        :param filters: Haystack filter dictionary.
        :returns: Equivalent Mem0 filter dictionary.
        """

        def _comparison(cond: dict[str, Any]) -> dict[str, Any]:
            op_map = {
                "==": lambda f, v: {f: v},
                "!=": lambda f, v: {f: {"ne": v}},
                ">": lambda f, v: {f: {"gt": v}},
                ">=": lambda f, v: {f: {"gte": v}},
                "<": lambda f, v: {f: {"lt": v}},
                "<=": lambda f, v: {f: {"lte": v}},
                "in": lambda f, v: {f: {"in": v if isinstance(v, list) else [v]}},
                "not in": lambda f, v: {f: {"ne": v}},
            }
            op = cond["operator"]
            if op not in op_map:
                msg = f"Unsupported filter operator: {op!r}"
                raise ValueError(msg)
            return op_map[op](cond["field"], cond["value"])

        def _logic(node: dict[str, Any]) -> dict[str, Any]:
            op = node["operator"].upper()
            if op not in ("AND", "OR", "NOT"):
                msg = f"Unsupported logic operator: {op!r}"
                raise ValueError(msg)
            return {op: [_convert(c) for c in node["conditions"]]}

        def _convert(node: dict[str, Any]) -> dict[str, Any]:
            return _comparison(node) if "field" in node else _logic(node)

        return _convert(filters)
