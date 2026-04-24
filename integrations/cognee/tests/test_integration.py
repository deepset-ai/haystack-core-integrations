# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import cognee
import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors.cognee import CogneeCognifier
from haystack_integrations.components.connectors.cognee._utils import run_sync
from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.components.writers.cognee import CogneeWriter
from haystack_integrations.memory_stores.cognee import CogneeMemoryStore

SKIP_REASON = "Export an env var called LLM_API_KEY containing the LLM API key to run this test."


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY", None),
    reason=SKIP_REASON,
)
@pytest.mark.integration
class TestCogneeMemoryStoreIntegration:
    def test_add_search_delete(self):
        store = CogneeMemoryStore(
            search_type="GRAPH_COMPLETION",
            top_k=3,
            dataset_name="haystack_integration_test",
        )

        store.delete_all_memories()

        messages = [
            ChatMessage.from_user("The capital of France is Paris."),
            ChatMessage.from_user("The Eiffel Tower is located in Paris."),
        ]
        store.add_memories(messages=messages)

        results = store.search_memories(query="What is the capital of France?", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ChatMessage) for r in results)

        store.delete_all_memories()


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY", None),
    reason=SKIP_REASON,
)
@pytest.mark.integration
class TestCogneeWriterRetrieverIntegration:
    def test_write_and_retrieve(self):
        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))

        writer = CogneeWriter(dataset_name="haystack_integration_test", auto_cognify=True)
        docs = [
            Document(content="Python is a programming language created by Guido van Rossum."),
            Document(content="Haystack is an open-source framework for building AI applications."),
        ]
        write_result = writer.run(documents=docs)
        assert write_result["documents_written"] == 2

        retriever = CogneeRetriever(
            memory_store=CogneeMemoryStore(
                search_type="GRAPH_COMPLETION",
                top_k=3,
                dataset_name="haystack_integration_test",
            ),
        )
        search_result = retriever.run(query="What is Haystack?")
        assert len(search_result["documents"]) > 0
        assert all(isinstance(d, Document) for d in search_result["documents"])

        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY", None),
    reason=SKIP_REASON,
)
@pytest.mark.integration
class TestCogneeCognifierIntegration:
    def test_write_then_cognify(self):
        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))

        writer = CogneeWriter(dataset_name="haystack_integration_test", auto_cognify=False)
        docs = [Document(content="Berlin is the capital of Germany.")]
        writer.run(documents=docs)

        cognifier = CogneeCognifier(dataset_name="haystack_integration_test")
        result = cognifier.run()
        assert result["cognified"] is True

        run_sync(cognee.prune.prune_data())
        run_sync(cognee.prune.prune_system(metadata=True))
