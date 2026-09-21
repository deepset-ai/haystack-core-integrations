# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import uuid

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.memory_stores.everos import EverOSMemoryStore


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EVEROS_CLOUD_API_KEY"), reason="Set EVEROS_CLOUD_API_KEY for a live EverOS Cloud test."
)
def test_live_everos_cloud_memory_round_trip():
    base_url = os.environ.get("EVEROS_TEST_BASE_URL", "https://api.evermind.ai")
    token = uuid.uuid4().hex[:12]
    user_id = f"haystack-integration-{token}"
    session_id = f"haystack-integration-{token}"
    marker = f"The integration marker is cobalt-{token}."
    store = EverOSMemoryStore(
        base_url=base_url,
        api_key=Secret.from_env_var("EVEROS_CLOUD_API_KEY"),
        timeout=30,
    )

    try:
        result = store.add_memories(
            messages=[ChatMessage.from_user(marker)],
            session_id=session_id,
            user_id=user_id,
            flush=True,
        )
        assert result["message_count"] == 1
        assert result["flush_status"]

        memories = []
        for _ in range(8):
            memories = store.search_memories(
                query=f"What is the integration marker ending in {token}?",
                user_id=user_id,
                session_id=session_id,
                include_profile=True,
                include_unprocessed=True,
            )
            if any(f"cobalt-{token}" in (memory.text or "") for memory in memories):
                break
            time.sleep(1.5)

        assert any(f"cobalt-{token}" in (memory.text or "") for memory in memories), [
            {
                "memory_type": memory.meta.get("everos", {}).get("memory_type"),
                "text": memory.text,
            }
            for memory in memories
        ]
    finally:
        store.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EVEROS_CLOUD_API_KEY"), reason="Set EVEROS_CLOUD_API_KEY for a live EverOS Cloud test."
)
def test_live_cloud_default_add_is_searchable_and_user_scoped():
    base_url = os.environ.get("EVEROS_TEST_BASE_URL", "https://api.evermind.ai")
    token = uuid.uuid4().hex[:12]
    user_id = f"haystack-default-add-{token}"
    other_user_id = f"haystack-other-user-{token}"
    session_id = f"haystack-default-add-{token}"
    marker = f"My cloud notebook color is saffron-{token}."
    store = EverOSMemoryStore(
        base_url=base_url,
        api_key=Secret.from_env_var("EVEROS_CLOUD_API_KEY"),
        timeout=30,
    )

    try:
        result = store.add_memories(
            messages=[ChatMessage.from_user(marker)],
            session_id=session_id,
            user_id=user_id,
        )
        assert result["message_count"] == 1

        memories = []
        for _ in range(8):
            memories = store.search_memories(
                query="What color is my cloud notebook?",
                user_id=user_id,
                method="hybrid",
                include_profile=True,
            )
            if any(f"saffron-{token}" in (memory.text or "") for memory in memories):
                break
            time.sleep(1.5)

        assert any(f"saffron-{token}" in (memory.text or "") for memory in memories), {
            "add_status": result["status"],
            "memories": [memory.text for memory in memories],
        }

        isolated = store.search_memories(
            query=f"saffron-{token}",
            user_id=other_user_id,
            method="keyword",
        )
        assert all(f"saffron-{token}" not in (memory.text or "") for memory in isolated)
    finally:
        store.close()
