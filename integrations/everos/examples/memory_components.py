"""Direct EverOS writer and retriever component example."""

import time

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.everos import EverOSMemoryRetriever
from haystack_integrations.components.writers.everos import EverOSMemoryWriter
from haystack_integrations.memory_stores.everos import EverOSMemoryStore


def main() -> None:
    """Write a short conversation, force cloud extraction, and query it."""
    store = EverOSMemoryStore()
    writer = EverOSMemoryWriter(memory_store=store, flush_on_write=True)
    retriever = EverOSMemoryRetriever(memory_store=store, top_k=5, include_profile=True)

    result = writer.run(
        [
            ChatMessage.from_user("My current prototype uses Qdrant, and I prefer concise Python examples."),
            ChatMessage.from_assistant("I'll remember the database choice and answer style."),
        ],
        session_id="everos-components-demo",
        user_id="alice",
        agent_id="haystack-docs-agent",
    )
    print(result)  # noqa: T201

    memories = []
    for attempt in range(6):
        memories = retriever.run("Which database and answer style does Alice prefer?", user_id="alice")["memories"]
        if memories:
            break
        time.sleep(2**attempt / 2)
    for memory in memories:
        print(memory.text)  # noqa: T201
    store.close()


if __name__ == "__main__":
    main()
