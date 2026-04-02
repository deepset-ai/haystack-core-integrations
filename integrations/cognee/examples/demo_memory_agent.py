#!/usr/bin/env python
"""
Demo: Cognee as a Memory Backend with User-Scoped Access

Shows how CogneeMemoryStore supports per-user memory isolation via user_id,
matching the MemoryStore protocol pattern used by Haystack's experimental Agent.

Two users (Alice and Bob) each store private memories. Then Alice creates a
shared dataset and grants Bob read access, demonstrating cross-user sharing.

Note: This demo uses cognee's user management APIs directly for setup (creating
users, granting permissions). These are admin operations outside the Haystack
integration's scope. In production, user management would typically be handled
by the application layer or cognee's API server.

Prerequisites:
    pip install -e "integrations/cognee"

Set your LLM API key:
    export LLM_API_KEY="sk-..."
"""

import asyncio

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


async def main():
    print("=== Cognee Memory Store — User Scoping Demo ===\n")

    # --- Setup: clean slate and create two users ---
    print("Setup: Pruning all data and creating users...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Re-create the database schema after pruning (prune_system drops all tables)
    await setup()

    alice = await create_user("alice@example.com", "password", is_verified=True)
    bob = await create_user("bob@example.com", "password", is_verified=True)
    alice_id = str(alice.id)
    bob_id = str(bob.id)
    print(f"  Created Alice (id={alice_id[:8]}...) and Bob (id={bob_id[:8]}...)\n")

    # =========================================================================
    # Part 1: Private memories — each user can only see their own
    # =========================================================================
    print("--- Part 1: Private memories (user isolation) ---\n")

    alice_store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", dataset_name="alice_notes")
    bob_store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", dataset_name="bob_notes")

    # Alice adds her private memories
    print("Alice adds memories to 'alice_notes'...")
    alice_store.add_memories(
        messages=[
            ChatMessage.from_user("The project deadline is next Friday."),
        ],
        user_id=alice_id,
    )
    print("  Done.\n")

    # Bob adds his private memories
    print("Bob adds memories to 'bob_notes'...")
    bob_store.add_memories(
        messages=[
            ChatMessage.from_user("The client meeting is on Wednesday at 2pm."),
            ChatMessage.from_user("The new API endpoint needs authentication."),
        ],
        user_id=bob_id,
    )
    print("  Done.\n")

    # Alice searches her own store — should find results
    print("Alice searches her own store for 'project deadline':")
    results = alice_store.search_memories(query="What is the project deadline?", user_id=alice_id)
    print(f"  Found {len(results)} result(s)")
    for i, msg in enumerate(results, 1):
        print(f"  [{i}] {msg.text}")
    print()

    # Bob searches his own store — should find results
    print("Bob searches his own store for 'client meeting':")
    results = bob_store.search_memories(query="When is the client meeting?", user_id=bob_id)
    print(f"  Found {len(results)} result(s)")
    for i, msg in enumerate(results, 1):
        print(f"  [{i}] {msg.text}")
    print()

    # Alice searches Bob's store — should find nothing (no permission)
    print("Alice tries to search Bob's store (no permission):")
    results = bob_store.search_memories(query="When is the client meeting?", user_id=alice_id)
    print(f"  Found {len(results)} result(s) — access is isolated!\n")

    # Bob searches Alice's store — should find nothing (no permission)
    print("Bob tries to search Alice's store (no permission):")
    results = alice_store.search_memories(query="What is the project deadline?", user_id=bob_id)
    print(f"  Found {len(results)} result(s) — access is isolated!\n")

    # =========================================================================
    # Part 2: Shared dataset — Alice creates it, grants Bob read access
    # =========================================================================
    print("--- Part 2: Shared dataset ---\n")

    shared_store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", dataset_name="team_shared")

    # Alice adds to the shared dataset (she becomes the owner)
    print("Alice adds memories to 'team_shared'...")
    shared_store.add_memories(
        messages=[
            ChatMessage.from_user("The team standup is every morning at 9am."),
            ChatMessage.from_user("Our tech stack is Python, Haystack, and Cognee."),
        ],
        user_id=alice_id,
    )
    print("  Done.\n")

    # Bob tries to search the shared store BEFORE getting permission — should find nothing
    print("Bob tries to search 'team_shared' BEFORE permission:")
    results = shared_store.search_memories(query="When is the team standup?", user_id=bob_id)
    print(f"  Found {len(results)} result(s) — no access yet.\n")

    # Grant Bob read permission on the shared dataset
    print("Alice grants Bob read access to 'team_shared'...")
    shared_datasets = await get_authorized_existing_datasets(["team_shared"], "read", alice)
    shared_dataset_id = shared_datasets[0].id
    await give_permission_on_dataset(bob, shared_dataset_id, "read")
    print("  Done.\n")

    # Bob searches via the MemoryStore — the store automatically resolves shared datasets
    print("Bob searches 'team_shared' AFTER getting read permission:")
    results = shared_store.search_memories(query="When is the team standup?", user_id=bob_id)
    print(f"  Found {len(results)} result(s)")
    for i, msg in enumerate(results, 1):
        print(f"  [{i}] {msg.text}")
    print()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
