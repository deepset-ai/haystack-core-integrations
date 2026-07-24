# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Dakera memory store for Haystack — self-hosted persistent vector memory."""

from __future__ import annotations

import os
from typing import Any

import httpx
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:3000"


class DakeraMemoryStore:
    """Persistent, decay-weighted vector memory backed by a self-hosted Dakera server.

    Dakera stores memories with access-weighted importance scoring: recent and
    frequently-accessed memories naturally surface higher than stale ones. The store
    is a thin REST client over the Dakera HTTP API (``POST /v1/memory/store`` to
    persist and ``POST /v1/memory/recall`` for decay-weighted semantic recall).

    Self-host with Docker (see https://github.com/dakera-ai/dakera-deploy):
        docker run -p 3000:3000 -e DAKERA_ROOT_API_KEY=your-key ghcr.io/dakera-ai/dakera:latest

    Args:
        base_url: Dakera server base URL. Defaults to the ``DAKERA_API_URL`` env var
            or ``http://localhost:3000`` (the server's default REST port).
        api_key: Dakera API key as a Haystack ``Secret``. Sent as an ``X-API-Key``
            header. Defaults to the ``DAKERA_API_KEY`` env var.
        default_agent_id: Agent namespace used to isolate memories. The Dakera API
            requires an ``agent_id`` on every call; this is the fallback. Defaults to "haystack".
        timeout: HTTP request timeout in seconds. Defaults to 10.

    Example:
        ```python
        from haystack.dataclasses import ChatMessage
        from haystack.utils import Secret
        from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

        store = DakeraMemoryStore(
            base_url="http://localhost:3000",
            api_key=Secret.from_env_var("DAKERA_API_KEY"),
        )
        store.store_memories([ChatMessage.from_user("Alice prefers concise answers.")])
        memories = store.recall_memories("How should I answer Alice?")
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: Secret | None = None,
        default_agent_id: str = "haystack",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("DAKERA_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or Secret.from_env_var("DAKERA_API_KEY", strict=False)
        self.default_agent_id = default_agent_id
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key.resolve_value() if self.api_key else None
        if key:
            # Dakera's default auth header; the server also accepts "Authorization: Bearer".
            headers["X-API-Key"] = key
        return headers

    def store_memories(
        self,
        messages: list[ChatMessage],
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist the text of each ``ChatMessage`` to Dakera via ``POST /v1/memory/store``.

        Args:
            messages: Chat messages whose text is stored as memories. Messages with no
                text content are skipped.
            agent_id: Agent namespace. Falls back to ``default_agent_id``.
            session_id: Optional session identifier grouping related memories.
            tags: Optional tags attached to each stored memory.
            metadata: Optional key-value metadata merged into each memory (per-message
                ``ChatMessage.meta`` is merged on top).

        Returns:
            Number of memories successfully stored.
        """
        stored = 0
        with httpx.Client(timeout=self.timeout) as client:
            for message in messages:
                content = message.text
                if not content:
                    continue
                merged_meta = {**(metadata or {}), **(message.meta or {})}
                payload: dict[str, Any] = {
                    "content": content,
                    "agent_id": agent_id or self.default_agent_id,
                }
                if session_id:
                    payload["session_id"] = session_id
                if tags:
                    payload["tags"] = tags
                if merged_meta:
                    payload["metadata"] = merged_meta
                try:
                    resp = client.post(
                        f"{self.base_url}/v1/memory/store",
                        headers=self._headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                    stored += 1
                except httpx.HTTPError as exc:
                    logger.warning("DakeraMemoryStore: failed to store memory: {error}", error=exc)
        return stored

    def recall_memories(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[ChatMessage]:
        """Recall memories relevant to ``query`` via decay-weighted semantic search.

        Calls ``POST /v1/memory/recall``. Each returned memory becomes a system-role
        ``ChatMessage`` whose ``meta`` carries the Dakera fields (``id``, ``score``,
        ``agent_id``, ``session_id``, ``importance``, ``tags``, ``created_at``) plus any
        user metadata stored with the memory.

        Args:
            query: Natural language query.
            agent_id: Agent namespace. Falls back to ``default_agent_id``.
            session_id: Optional session filter.
            tags: Optional tag filter.
            top_k: Maximum number of memories to return.

        Returns:
            A list of ``ChatMessage`` objects, most relevant first. Empty on error.
        """
        payload: dict[str, Any] = {
            "query": query,
            "agent_id": agent_id or self.default_agent_id,
            "top_k": top_k,
        }
        if session_id:
            payload["session_id"] = session_id
        if tags:
            payload["tags"] = tags
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/memory/recall",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("DakeraMemoryStore: recall failed: {error}", error=exc)
            return []
        return [self._to_chat_message(item) for item in data.get("memories", [])]

    @staticmethod
    def _to_chat_message(result: dict[str, Any]) -> ChatMessage:
        """Convert a Dakera recall result (``{memory: {...}, score}``) to a ChatMessage."""
        memory = result.get("memory") or {}
        user_metadata = memory.get("metadata") or {}
        meta: dict[str, Any] = {
            "id": memory.get("id"),
            "score": result.get("score"),
            "agent_id": memory.get("agent_id"),
            "session_id": memory.get("session_id"),
            "importance": memory.get("importance"),
            "tags": memory.get("tags"),
            "created_at": memory.get("created_at"),
            **user_metadata,
        }
        return ChatMessage.from_system(text=memory.get("content", ""), meta=meta)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            base_url=self.base_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            default_agent_id=self.default_agent_id,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DakeraMemoryStore:
        if api_key := data.get("init_parameters", {}).get("api_key"):
            data["init_parameters"]["api_key"] = Secret.from_dict(api_key)
        return default_from_dict(cls, data)
