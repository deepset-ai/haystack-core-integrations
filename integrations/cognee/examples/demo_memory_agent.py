#!/usr/bin/env python
"""
Minimal cognee-haystack demo — persistent vs session memory in four phases.

Phase 1 — `persistent_writer` (no `session_id`) seeds long-lived facts into
          cognee's permanent knowledge graph.
Phase 2 — `session_writer` (`session_id=...`) seeds session-only context into
          cognee's session cache. The graph itself doesn't change.
Phase 3 — Agent loop: `CogneeRetriever` calls `cognee.recall(query, session_id=...)`
          which auto-captures each turn as a QA entry in the session. **No
          CogneeWriter in the pipeline** — cognee's recall is the session-write
          path per the docs.
Phase 4 — `chat_store.improve()` promotes the session into the permanent
          graph via `cognee.improve(dataset=..., session_ids=[...])`.

Environment (loaded from repo-root .env):
    LLM_API_KEY        Required. cognee's LLM provider key.
    EMBEDDING_API_KEY  Optional; defaults to LLM_API_KEY when unset.
    OPENAI_API_KEY     Required. Used by Haystack's OpenAIChatGenerator.

Run:
    cd integrations/cognee
    .venv/bin/hatch run test:python examples/demo_memory_agent.py
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("LOG_LEVEL", "WARNING")

# Load .env from the repo root before cognee imports read any config.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

import cognee
from cognee.api.v1.visualize import visualize_multi_user_graph
from cognee.modules.users.methods import get_default_user
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.components.writers.cognee import CogneeWriter
from haystack_integrations.memory_stores.cognee import CogneeMemoryStore

# cognee binds its graph engine to whichever event loop touches it first.
# Route the demo's direct cognee calls through the integration's background
# loop so reads and writes share state.
from haystack_integrations.memory_stores.cognee.memory_store import _run_sync

logging.basicConfig(level=logging.WARNING)

DATASET = "agent_memory_minimal"
SESSION = "alice_chat_42"

ARTIFACTS = Path(__file__).resolve().parent / "graph_snapshots"
ARTIFACTS.mkdir(exist_ok=True)

# Long-lived facts
PERSISTENT_MEMORIES = [
    "My name is Alice. I'm a senior data scientist at Acme Corp specialising in NLP and knowledge graphs.",
    "My current project is building an internal documentation search system powered by Haystack and Cognee.",
    "My team: Bob is the ML engineer and Carol handles infrastructure.",
    "I prefer concise answers with Python code examples over long prose explanations.",
]

# Session-only context
SESSION_MEMORIES = [
    "Bob is having trouble with the new documentation search system.",
    "Carol helps Bob troubleshoot the issue.",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to persistent memory of past conversations. "
    "Any system messages at the start of the conversation contain relevant memories. "
    "Be concise; prefer short answers and Python code examples."
)


async def _visualize_all_datasets(destination_file_path: str) -> None:
    """Render a combined graph across every dataset the default user can read.

    Uses `cognee.visualize_multi_user_graph` with explicit `(user, dataset)` pairs
    so each dataset's graph is read inside its own database context. Works whether
    or not `ENABLE_BACKEND_ACCESS_CONTROL` is enabled.
    """
    user = await get_default_user()
    datasets = await cognee.datasets.list_datasets(user=user)
    pairs = [(user, ds) for ds in datasets]
    await visualize_multi_user_graph(pairs, destination_file_path=destination_file_path)


def build_pipeline(chat_store: CogneeMemoryStore) -> Pipeline:
    """Retriever → injector → agent. No writer: cognee.recall auto-captures the session QA."""
    pipeline = Pipeline()
    pipeline.add_component("retriever", CogneeRetriever(memory_store=chat_store))
    pipeline.add_component(
        "injector",
        OutputAdapter(
            template="{{ memories + user_messages }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )
    pipeline.add_component(
        "agent",
        Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=SYSTEM_PROMPT,
        ),
    )
    pipeline.connect("retriever.messages", "injector.memories")
    pipeline.connect("injector.output", "agent.messages")
    return pipeline


async def main() -> None:
    # Direct `cognee.forget(everything=True)` instead of `store.delete_all_memories()`:
    # the protocol method is dataset-scoped and leaves the session cache alone, but at
    # demo start we want a global wipe including any leftover session entries.
    print("Forgetting previous data for a clean start...")
    _run_sync(cognee.forget(everything=True))
    print("Done.\n")

    # One store, two writers — `session_id` on the writer picks the tier.
    # `self_improvement=False` keeps Phase 4's `improve()` as the only improve
    # trigger; otherwise cognee runs improve per write and we'd see duplicated
    # nodes after the explicit improve.
    seed_store = CogneeMemoryStore(dataset_name=DATASET, self_improvement=False)

    # ─── Phase 1: persistent seed (writer has no session_id) ───────────────────
    print(f"Phase 1: persistent_writer -> permanent graph ({len(PERSISTENT_MEMORIES)} facts)...")
    persistent_writer = CogneeWriter(memory_store=seed_store)
    persistent_writer.run(messages=[ChatMessage.from_user(fact) for fact in PERSISTENT_MEMORIES])

    snapshot_1 = ARTIFACTS / "1_after_persistent_seed.html"
    _run_sync(_visualize_all_datasets(str(snapshot_1)))
    print(f"  Graph snapshot -> {snapshot_1}\n")

    # ─── Phase 2: session seed (writer has session_id set) ─────────────────────
    print(f"Phase 2: session_writer    -> session cache  ({len(SESSION_MEMORIES)} facts)...")
    session_writer = CogneeWriter(memory_store=seed_store, session_id=SESSION)
    session_writer.run(messages=[ChatMessage.from_user(fact) for fact in SESSION_MEMORIES])

    snapshot_2 = ARTIFACTS / "2_after_session_seed.html"
    _run_sync(_visualize_all_datasets(str(snapshot_2)))
    print(f"  Graph snapshot -> {snapshot_2}")
    print("  (Session writes don't touch the graph — should look like snapshot 1.)\n")

    # ─── Phase 3: agent loop (no writer; cognee.recall auto-captures the session QA) ──
    print("Phase 3: agent loop (retriever -> injector -> agent)\n")
    # Session-scoped store so the retriever's recall is session-aware.
    chat_store = CogneeMemoryStore(dataset_name=DATASET, session_id=SESSION)
    pipeline = build_pipeline(chat_store)

    turns = [
        "Hi! Can you remind me what project I'm currently working on?",
        "What's the tech stack we're using for it?",
        "Who else is on my team, and what are their roles?",
        "Based on what you know about me, give me a quick tip for structuring a new Haystack pipeline component.",
    ]
    for user_text in turns:
        print(f"User:  {user_text}")
        result = pipeline.run(
            {
                "retriever": {"query": user_text},
                "injector": {"user_messages": [ChatMessage.from_user(user_text)]},
            }
        )
        reply = result["agent"]["last_message"].text or "(no reply)"
        print(f"Agent: {reply}\n")

    snapshot_3 = ARTIFACTS / "3_after_chat.html"
    _run_sync(_visualize_all_datasets(str(snapshot_3)))
    print(f"  Graph snapshot -> {snapshot_3}")
    print("  (Still graph-unchanged: session writes from recall live in the cache.)\n")

    print("--- Session cache contents (cognee.session.get_session) ---")
    entries = _run_sync(cognee.session.get_session(session_id=SESSION))
    print(f"{len(entries)} entries in session {SESSION!r}")
    for i, e in enumerate(entries, 1):
        print(f"\n[{i}] qa_id={e.qa_id} time={e.time}")
        print(f"    question: {e.question!r}")
        print(f"    answer  : {e.answer!r}")

    # ─── Phase 4: improve session -> permanent graph ──────────────────────────
    print(f"\nPhase 4: chat_store.improve() -> cognee.improve(dataset={DATASET!r}, session_ids=[{SESSION!r}])...")
    chat_store.improve()

    snapshot_4 = ARTIFACTS / "4_after_improve.html"
    _run_sync(_visualize_all_datasets(str(snapshot_4)))
    print(f"  Graph snapshot -> {snapshot_4}")
    print("  (Graph now includes session-derived nodes.)")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
