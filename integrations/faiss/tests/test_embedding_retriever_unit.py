# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore


@pytest.fixture
def store() -> FAISSDocumentStore:
    store = FAISSDocumentStore(embedding_dim=3)
    store.write_documents(
        [
            Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0], meta={"kind": "book"}),
            Document(id="b", content="beta", embedding=[0.0, 1.0, 0.0], meta={"kind": "paper"}),
        ]
    )
    return store


class TestInit:
    def test_defaults(self, store):
        retriever = FAISSEmbeddingRetriever(document_store=store)

        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.filter_policy == FilterPolicy.REPLACE


class TestRun:
    def test_returns_the_documents_nearest_to_the_query(self, store):
        retriever = FAISSEmbeddingRetriever(document_store=store, top_k=1)

        documents = retriever.run(query_embedding=[1.0, 0.0, 0.0])["documents"]

        assert [doc.id for doc in documents] == ["a"]


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_matches_the_synchronous_result(self, store):
        retriever = FAISSEmbeddingRetriever(document_store=store, top_k=1)

        documents = (await retriever.run_async(query_embedding=[1.0, 0.0, 0.0]))["documents"]

        assert [doc.id for doc in documents] == [
            doc.id for doc in retriever.run(query_embedding=[1.0, 0.0, 0.0])["documents"]
        ]


class TestSerialization:
    def test_to_dict_and_from_dict_round_trip(self, store):
        retriever = FAISSEmbeddingRetriever(
            document_store=store,
            filters={"field": "meta.kind", "operator": "==", "value": "book"},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
        )

        data = retriever.to_dict()
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filter_policy"] == "merge"
        assert data["init_parameters"]["document_store"]["init_parameters"]["embedding_dim"] == 3

        deserialized = FAISSEmbeddingRetriever.from_dict(data)

        assert deserialized.top_k == 5
        assert deserialized.filter_policy == FilterPolicy.MERGE
        assert deserialized.filters == {"field": "meta.kind", "operator": "==", "value": "book"}
        assert isinstance(deserialized.document_store, FAISSDocumentStore)
