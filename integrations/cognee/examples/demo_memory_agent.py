#!/usr/bin/env python
"""
Demo: Cognee as a Memory Backend for Haystack's Agent

Shows how to use CogneeMemoryStore with Haystack's experimental Agent
to give conversational memory backed by Cognee's knowledge engine.

Prerequisites:
    pip install -e "integrations/cognee[memory]"

Set your LLM API key:
    export LLM_API_KEY="sk-..."
"""

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors.cognee import CogneeMemoryStore
from cognee.api.v1.visualize.visualize import visualize_graph
import asyncio
import os

async def main():
    print("=== Cognee Memory Store Demo ===\n")

    store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", top_k=5, dataset_name="agent_memory")

    # --- Step 1: Clear any previous memories ---
    print("1. Clearing previous memories...")
    store.delete_all_memories()
    print("   Done.\n")

    # --- Step 2: Add some memories ---
    print("2. Adding memories...")
    messages = [
        ChatMessage.from_user(
            "My name is Alice and I'm working on the Cognee-Haystack integration. "
            "The deadline is next Friday."
        ),
        ChatMessage.from_user(
            "We decided to use GRAPH_COMPLETION as the default search type "
            "because it retrieves the most relevant memories by graph traversal and vector search"
        ),
        ChatMessage.from_assistant(
            "I'll remember that the deadline is next Friday and that GRAPH_COMPLETION "
            "is the preferred search type for this project."
        ),
    ]
    store.add_memories(messages=messages)
    print(f"   Added {len(messages)} messages as memories.\n")

    visualization_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "demo_memory_agent.html"
    )
    await visualize_graph(visualization_path)

    # --- Step 3: Search memories ---
    queries = [
        "What is the project deadline?",
        "Which search type should we use?",
        "Who is working on the integration?",
    ]

    for query in queries:
        print(f"3. Searching memories: '{query}'")
        results = store.search_memories(query=query, top_k=3)
        print(f"   Found {len(results)} memory(ies):")
        for i, msg in enumerate(results, 1):
            print(f"   [{i}] {msg.text}")
        print()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
