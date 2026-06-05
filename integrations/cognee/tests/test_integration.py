# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import uuid

import cognee  # type: ignore[import-untyped]
import pytest
from cognee.modules.data.exceptions import DatasetNotFoundError  # type: ignore[import-untyped]
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.components.writers.cognee import CogneeWriter
from haystack_integrations.memory_stores.cognee import CogneeMemoryStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("LLM_API_KEY"),
        reason="Set LLM_API_KEY (cognee's LLM provider key) to run cognee integration tests.",
    ),
]


@pytest.fixture
def dataset_name() -> str:
    # Unique per-run so concurrent CI shards don't collide.
    return f"haystack_it_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _cleanup(dataset_name: str):
    yield
    # Best-effort — dataset may already be gone.
    with contextlib.suppress(Exception):
        asyncio.run(cognee.forget(dataset=dataset_name))


class TestCogneeIntegration:
    def test_remember_then_recall(self, dataset_name: str):
        store = CogneeMemoryStore(dataset_name=dataset_name, search_type="GRAPH_COMPLETION")
        store.add_memories(messages=[ChatMessage.from_user("Marie Curie discovered radium in 1898.")])

        results = store.search_memories(query="Who discovered radium?")

        assert results, "expected at least one memory from cognee"
        assert any("curie" in m.text.lower() or "radium" in m.text.lower() for m in results)

    def test_writer_then_retriever_pipeline(self, dataset_name: str):
        store = CogneeMemoryStore(dataset_name=dataset_name)
        writer = CogneeWriter(memory_store=store)
        retriever = CogneeRetriever(memory_store=store, top_k=3)

        writer.run(messages=[ChatMessage.from_user("Ada Lovelace wrote the first computer program.")])
        out = retriever.run(query="Who wrote the first computer program?")

        assert out["messages"], "retriever returned no messages"
        assert any("lovelace" in m.text.lower() or "program" in m.text.lower() for m in out["messages"])

    def test_search_after_forget_raises(self, dataset_name: str):
        """Forget actually deletes the dataset (recall against it raises DatasetNotFoundError)."""
        store = CogneeMemoryStore(dataset_name=dataset_name)
        store.add_memories(messages=[ChatMessage.from_user("Marie Curie discovered radium in 1898.")])
        store.delete_all_memories()

        with pytest.raises(DatasetNotFoundError):
            store.search_memories(query="radium")
