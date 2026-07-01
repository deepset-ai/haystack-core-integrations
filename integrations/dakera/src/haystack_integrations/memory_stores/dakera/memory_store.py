# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Dakera memory store for Haystack — self-hosted persistent vector memory."""

from __future__ import annotations

import os
from typing import Any

import httpx
from haystack import default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


class DakeraMemoryStore:
    """Persistent, decay-weighted vector memory backed by a self-hosted Dakera server.

    Dakera stores memories with access-weighted importance scoring: recent and
    frequently-accessed memories naturally surface higher than stale ones.

    Self-host via Docker:
        docker run -p 3300:3300 -e DAKERA_API_KEY=your-key ghcr.io/dakera-ai/dakera:latest

    Args:
        base_url: Dakera server base URL. Defaults to DAKERA_API_URL env var or http://localhost:3300.
        api_key: Dakera API key as a Secret. Defaults to DAKERA_API_KEY env var.
        default_agent_id: Default agent/namespace for memory isolation. Defaults to "haystack".
        timeout: HTTP request timeout in seconds. Defaults to 10.

    Example:
        .. code-block:: python

            from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

            store = DakeraMemoryStore(
                base_url="http://localhost:3300",
                api_key=Secret.from_env_var("DAKERA_API_KEY"),
            )
            store.store_memories(
                messages=["User asked about Python memory management"],
                user_id="alice",
            )
            results = store.search_memories(query="Python memory", user_id="alice")
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: Secret | None = None,
        default_agent_id: str = "haystack",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("DAKERA_API_URL", "http://localhost:3300")).rstrip("/")
        self.api_key = api_key or Secret.from_env_var("DAKERA_API_KEY", strict=False)
        self.default_agent_id = default_agent_id
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        key = self.api_key.resolve_value() if self.api_key else None
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def store_memories(
        self,
        messages: list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist a list of messages to Dakera.

        Args:
            messages: Text content to store.
            user_id: Optional user identifier.
            agent_id: Agent/namespace. Falls back to default_agent_id.
            session_id: Optional session identifier.
            metadata: Optional key-value metadata attached to each memory.

        Returns:
            Number of messages successfully stored.
        """
        stored = 0
        with httpx.Client(timeout=self.timeout) as client:
            for msg in messages:
                try:
                    resp = client.post(
                        f"{self.base_url}/v1/memories",
                        headers=self._headers(),
                        json={
                            "content": msg,
                            "agent_id": agent_id or self.default_agent_id,
                            **({"session_id": session_id} if session_id else {}),
                            **({"metadata": {**(metadata or {}), **({"user_id": user_id} if user_id else {})}} if user_id or metadata else {}),
                        },
                    )
                    resp.raise_for_status()
                    stored += 1
                except httpx.HTTPError as exc:
                    logger.warning("DakeraMemoryStore: failed to store memory: %s", exc)
        return stored

    def search_memories(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for memories relevant to the query using decay-weighted semantic search.

        Args:
            query: Natural language query.
            user_id: Optional user filter.
            agent_id: Agent/namespace. Falls back to default_agent_id.
            session_id: Optional session filter.
            top_k: Number of results to return.

        Returns:
            List of memory dicts with keys: content, score, id, metadata.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/memories/search",
                    headers=self._headers(),
                    json={
                        "query": query,
                        "agent_id": agent_id or self.default_agent_id,
                        "top_k": top_k,
                        **({"session_id": session_id} if session_id else {}),
                        **({"filter": {"metadata.user_id": user_id}} if user_id else {}),
                    },
                )
                resp.raise_for_status()
                return resp.json().get("results", [])
        except httpx.HTTPError as exc:
            logger.warning("DakeraMemoryStore: search failed: %s", exc)
            return []

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            base_url=self.base_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            default_agent_id=self.default_agent_id,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DakeraMemoryStore":
        if api_key := data.get("init_parameters", {}).get("api_key"):
            data["init_parameters"]["api_key"] = Secret.from_dict(api_key)
        return default_from_dict(cls, data)
