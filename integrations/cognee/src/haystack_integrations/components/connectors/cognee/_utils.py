# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar
from uuid import UUID

from cognee.modules.users.methods import get_user  # type: ignore[import-untyped]

T = TypeVar("T")

# Persistent background event loop for run_sync when called from an async context.
# A single loop is reused so that cognee's internal asyncio.Lock objects stay bound
# to one event loop across multiple run_sync calls.
_background_loop: asyncio.AbstractEventLoop | None = None
_background_thread: threading.Thread | None = None
_lock = threading.Lock()

CogneeSearchType = Literal[
    "GRAPH_COMPLETION",
    "RAG_COMPLETION",
    "CHUNKS",
    "CHUNKS_LEXICAL",
    "SUMMARIES",
    "TRIPLET_COMPLETION",
    "GRAPH_SUMMARY_COMPLETION",
    "GRAPH_COMPLETION_COT",
    "GRAPH_COMPLETION_CONTEXT_EXTENSION",
    "CYPHER",
    "NATURAL_LANGUAGE",
    "TEMPORAL",
    "CODING_RULES",
    "FEELING_LUCKY",
]


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background event loop running in a daemon thread."""
    global _background_loop, _background_thread  # noqa: PLW0603

    with _lock:
        if _background_loop is None or _background_loop.is_closed():
            _background_loop = asyncio.new_event_loop()
            _background_thread = threading.Thread(target=_background_loop.run_forever, daemon=True)
            _background_thread.start()
    return _background_loop


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from a synchronous context.

    If no event loop is running, uses asyncio.run() directly.
    If already inside an async context, submits the coroutine to a persistent
    background event loop so that cognee's internal asyncio.Lock objects remain
    bound to a single loop across calls.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already in an async context — submit to the persistent background loop
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def extract_text(item: Any) -> str:
    """
    Best-effort text extraction from a Cognee search result item.

    Cognee's search results are not a single public type — depending on the
    search strategy, an item can be one of three shapes (all cognee-internal):

    1. A plain `str` — returned by LLM-completion search types
       (e.g. `GRAPH_COMPLETION`, `RAG_COMPLETION`).
    2. A `dict` — returned by structured search types; the text is read from
       one of the `content` / `text` / `description` / `name` keys.
    3. A cognee model object — e.g. a `DataPoint` subclass such as
       `TextChunk` or `EntityType`, carrying the same attributes.

    Since these types are internal to cognee and not part of its public API,
    `item` is typed as `Any` and this function probes for known shapes rather
    than narrowing the type.

    :param item: A single element returned by `cognee.search(...)`.
    :returns: Best-effort extracted text, falling back to `str(item)`.
    """
    if isinstance(item, str):
        return item

    for attr in ("content", "text", "description", "name"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if val and isinstance(val, str):
                return val

    if isinstance(item, dict):
        for key in ("content", "text", "description", "name"):
            if key in item and isinstance(item[key], str):
                return item[key]

    return str(item)


async def _get_cognee_user(user_id: str) -> Any:
    """
    Resolve a user_id string to a cognee User object.

    Converts the given UUID string to a cognee User via ``cognee.modules.users.methods.get_user``.

    :param user_id: UUID string identifying the cognee user.
    :returns: A cognee ``User`` object.
    :raises ValueError: If user_id is not a valid UUID or the user is not found.
    """
    try:
        uid = UUID(user_id)
    except ValueError as e:
        msg = f"Invalid user_id: '{user_id}' is not a valid UUID."
        raise ValueError(msg) from e

    return await get_user(uid)
