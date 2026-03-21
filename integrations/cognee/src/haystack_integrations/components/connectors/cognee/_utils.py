# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypeVar

T = TypeVar("T")

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


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from a synchronous context.

    If no event loop is running, uses asyncio.run() directly.
    If already inside an async context, runs the coroutine in a separate thread
    to avoid blocking the existing event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already in an async context — run in a thread to avoid nested loop issues
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def extract_text(item: Any) -> str:
    """Best-effort text extraction from a Cognee search result item."""
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
