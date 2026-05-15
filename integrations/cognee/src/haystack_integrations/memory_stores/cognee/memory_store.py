# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar
from uuid import UUID

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

import cognee  # type: ignore[import-untyped]
from cognee.modules.users.methods import get_user  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

T = TypeVar("T")

CogneeSearchType = Literal[
    "SUMMARIES",
    "CHUNKS",
    "RAG_COMPLETION",
    "TRIPLET_COMPLETION",
    "GRAPH_COMPLETION",
    "GRAPH_COMPLETION_DECOMPOSITION",
    "GRAPH_SUMMARY_COMPLETION",
    "CYPHER",
    "NATURAL_LANGUAGE",
    "GRAPH_COMPLETION_COT",
    "GRAPH_COMPLETION_CONTEXT_EXTENSION",
    "FEELING_LUCKY",
    "TEMPORAL",
    "CODING_RULES",
    "CHUNKS_LEXICAL",
]


# Persistent background loop reused across calls when the caller is already
# inside an event loop. cognee creates asyncio.Lock objects bound to the loop
# they're first awaited on, so a fresh loop per call would later raise
# "lock bound to a different loop" — hence one shared loop.
_background_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    global _background_loop  # noqa: PLW0603
    with _loop_lock:
        if _background_loop is None or _background_loop.is_closed():
            _background_loop = asyncio.new_event_loop()
            threading.Thread(target=_background_loop.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _background_loop).result(timeout=300)


async def _resolve_user(user_id: str | None) -> Any:
    if user_id is None:
        return None
    return await get_user(UUID(user_id))


def _render(item: Any) -> str | None:
    # cognee.recall returns a discriminated RecallResponse union — pick the right
    # text field per source rather than probing attributes.
    source = getattr(item, "source", None)
    if source == "graph":
        return item.text
    if source == "session":
        return item.answer or item.question or None
    if source == "graph_context":
        return item.content
    if source == "trace":
        return item.memory_context or None
    return None


class CogneeMemoryStore:
    """
    Memory backend backed by Cognee, implementing the haystack-experimental `MemoryStore` protocol.

    Wraps cognee's V2 memory API: `add_memories` -> `cognee.remember`,
    `search_memories` -> `cognee.recall`, `improve` -> `cognee.improve`,
    `delete_all_memories` -> `cognee.forget`.

    `session_id` selects the tier — set it to use cognee's session cache (cheap,
    no LLM extraction, session-aware recall); leave `None` for the permanent
    graph.

    `self_improvement` is forwarded to `cognee.remember` and defaults to `True`
    (same as cognee). On the permanent tier it awaits `improve` inline; on the
    session tier it schedules `improve` as a fire-and-forget background task.
    Set to `False` when you want `improve()` to be the only improve trigger
    — otherwise an explicit `improve()` runs improve twice and produces
    near-duplicate graph nodes.
    """

    def __init__(
        self,
        *,
        search_type: CogneeSearchType = "GRAPH_COMPLETION",
        top_k: int = 5,
        dataset_name: str = "haystack_memory",
        session_id: str | None = None,
        self_improvement: bool = True,
    ):
        """
        Initialize the store.

        :param search_type: Cognee search strategy used by `search_memories`.
        :param top_k: Default max results for `search_memories`.
        :param dataset_name: Cognee dataset backing this store.
        :param session_id: When set, use the session-cache tier; otherwise the permanent graph.
        :param self_improvement: Forwarded to `cognee.remember` (default `True`, matches cognee).
            Set to `False` when `improve()` should be the only improve trigger.
        """
        self.search_type = search_type
        self.top_k = top_k
        self.dataset_name = dataset_name
        self.session_id = session_id
        self.self_improvement = self_improvement

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Persist messages via `cognee.remember`.

        Permanent tier batches all texts into one call; session tier writes one
        entry per message (matches cognee's session example). Empty messages
        are skipped.

        :param messages: Messages to store.
        :param user_id: Cognee user UUID; `None` uses cognee's default user.
        :param session_id: Per-call override of the store's `session_id`.
        """
        texts = [m.text for m in messages if m.text]
        if not texts:
            return

        target_session = session_id if session_id is not None else self.session_id

        async def _store() -> None:
            user = await _resolve_user(user_id)
            if target_session is not None:
                for text in texts:
                    await cognee.remember(
                        text,
                        dataset_name=self.dataset_name,
                        session_id=target_session,
                        user=user,
                        self_improvement=self.self_improvement,
                    )
            else:
                await cognee.remember(
                    texts,
                    dataset_name=self.dataset_name,
                    user=user,
                    self_improvement=self.self_improvement,
                )

        _run_sync(_store())
        logger.info(
            "Stored {n} memories in '{ds}' (session={s})",
            n=len(texts),
            ds=self.dataset_name,
            s=target_session,
        )

    def search_memories(
        self,
        *,
        query: str | None = None,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        """
        Search via `cognee.recall` and wrap each hit in a system `ChatMessage`.

        :param query: Natural-language query. Empty/`None` returns `[]`.
        :param top_k: Per-call override of the store's default.
        :param user_id: Cognee user UUID; `None` uses cognee's default user.
        """
        if not query:
            return []

        async def _search() -> list[Any]:
            user = await _resolve_user(user_id)
            return await cognee.recall(
                query,
                query_type=cognee.SearchType[self.search_type],
                datasets=[self.dataset_name],
                top_k=top_k if top_k is not None else self.top_k,
                session_id=self.session_id,
                user=user,
            )

        results = _run_sync(_search()) or []
        memories = [ChatMessage.from_system(text) for item in results if (text := _render(item))]
        logger.info("Found {n} memories for '{q}'", n=len(memories), q=query[:80])
        return memories

    def improve(self, *, session_id: str | None = None, user_id: str | None = None) -> None:
        """
        Promote session-cache content into the permanent graph via `cognee.improve`.

        Without any session_id this is a plain graph-enrichment pass.

        :param session_id: Session to promote; defaults to the store's `session_id`.
        :param user_id: Cognee user UUID; `None` uses cognee's default user.
        """
        target_session = session_id or self.session_id

        async def _improve() -> None:
            user = await _resolve_user(user_id)
            await cognee.improve(
                dataset=self.dataset_name,
                session_ids=[target_session] if target_session else None,
                user=user,
            )

        _run_sync(_improve())
        logger.info("Improved '{ds}' (session={s})", ds=self.dataset_name, s=target_session)

    def delete_all_memories(self, *, user_id: str | None = None) -> None:
        """
        Delete this dataset via `cognee.forget(dataset=...)`.

        Session cache survives (sessions aren't dataset-scoped) — use
        `cognee.forget(everything=True)` for a full wipe.
        """

        async def _delete() -> None:
            user = await _resolve_user(user_id)
            await cognee.forget(dataset=self.dataset_name, user=user)

        _run_sync(_delete())
        logger.info("Deleted '{ds}'", ds=self.dataset_name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this store for pipeline persistence."""
        return default_to_dict(
            self,
            search_type=self.search_type,
            top_k=self.top_k,
            dataset_name=self.dataset_name,
            session_id=self.session_id,
            self_improvement=self.self_improvement,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeMemoryStore":
        """Deserialize a store from a dict produced by `to_dict`."""
        return default_from_dict(cls, data)
