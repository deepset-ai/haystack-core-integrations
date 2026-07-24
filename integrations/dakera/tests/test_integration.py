# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests against a live Dakera server.

These are skipped unless ``DAKERA_API_URL`` points at a running Dakera instance
(see https://github.com/dakera-ai/dakera-deploy). To run locally:

    docker run -d -p 3000:3000 -e DAKERA_ROOT_API_KEY=demo ghcr.io/dakera-ai/dakera:latest
    export DAKERA_API_URL=http://localhost:3000 DAKERA_API_KEY=demo
    hatch run test:integration
"""

import os

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DAKERA_API_URL"),
        reason="requires a live Dakera server (set DAKERA_API_URL, e.g. http://localhost:3000)",
    ),
]


@pytest.fixture
def store():
    return DakeraMemoryStore(
        base_url=os.environ["DAKERA_API_URL"],
        api_key=Secret.from_env_var("DAKERA_API_KEY", strict=False),
        default_agent_id="haystack-integration-test",
    )


def test_store_and_recall_round_trip(store):
    written = store.store_memories(
        [ChatMessage.from_user("The Eiffel Tower is in Paris.")],
        session_id="itest",
        tags=["geography"],
    )
    assert written == 1

    messages = store.recall_memories("Where is the Eiffel Tower?", session_id="itest", top_k=5)
    assert isinstance(messages, list)
    assert all(isinstance(m, ChatMessage) for m in messages)
    assert any("Paris" in (m.text or "") for m in messages)


def test_components_round_trip(store):
    writer = DakeraMemoryWriter(memory_store=store)
    retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)

    writer.run(messages=[ChatMessage.from_user("Bob's favorite language is Rust.")], session_id="itest2")
    result = retriever.run(query="What is Bob's favorite language?", session_id="itest2")
    assert "memories" in result
    assert all(isinstance(m, ChatMessage) for m in result["memories"])
