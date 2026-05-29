# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSparseEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        ElasticsearchSparseEmbeddingRetriever(document_store=mock_store, filter_policy="keep")


def test_init_wrong_document_store_type():
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchSparseEmbeddingRetriever(document_store=Mock())


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=document_store)
    retriever_type = (
        "haystack_integrations.components.retrievers.elasticsearch."
        "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
    )
    assert retriever.to_dict() == {
        "type": retriever_type,
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "ELASTIC_API_KEY",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "api_key_id": {
                        "env_vars": [
                            "ELASTIC_API_KEY_ID",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "hosts": "some fake host",
                    "custom_mapping": None,
                    "index": "default",
                    "embedding_similarity_function": "cosine",
                    "sparse_vector_field": "sparse_vec",
                    "ingest_pipeline": None,
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }
    retriever = ElasticsearchSparseEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "sparse_embedding_retriever.ElasticsearchSparseEmbeddingRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = ElasticsearchSparseEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = [Document(content="Test doc")]
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = retriever.run(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store)
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = await retriever.run_async(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval_async.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


def test_run_init_params():
    """Init-time filters and top_k are passed through when no runtime params are provided."""
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = [Document(content="test document")]
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    retriever = ElasticsearchSparseEmbeddingRetriever(
        document_store=mock_store,
        filters=init_filters,
        top_k=3,
        filter_policy=FilterPolicy.MERGE,
    )
    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[0.5, 0.7])
    res = retriever.run(query_sparse_embedding=query_sparse_embedding)
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=query_sparse_embedding,
        filters=init_filters,
        top_k=3,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


def test_run_replace_filter_policy():
    """Runtime filter replaces init filter under REPLACE policy."""
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = []
    retriever = ElasticsearchSparseEmbeddingRetriever(
        document_store=mock_store,
        filters={"field": "meta.source", "operator": "==", "value": "wiki"},
        top_k=5,
        filter_policy=FilterPolicy.REPLACE,
    )
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}
    retriever.run(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        filters=runtime_filters,
    )
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        filters=runtime_filters,
        top_k=5,
    )


def test_run_merge_filter_policy():
    """Runtime filter is AND-combined with init filter under MERGE policy."""
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = []
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}
    retriever = ElasticsearchSparseEmbeddingRetriever(
        document_store=mock_store,
        filters=init_filters,
        top_k=5,
        filter_policy=FilterPolicy.MERGE,
    )
    retriever.run(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        filters=runtime_filters,
    )
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        filters={"operator": "AND", "conditions": [init_filters, runtime_filters]},
        top_k=5,
    )


def test_run_runtime_top_k_overrides():
    """top_k passed at run time overrides the init-time default."""
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval.return_value = []
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=mock_store, top_k=10)
    retriever.run(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        top_k=2,
    )
    mock_store._sparse_vector_retrieval.assert_called_once_with(
        query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.7]),
        filters={},
        top_k=2,
    )


@pytest.mark.integration
class TestElasticsearchSparseEmbeddingRetrieverIntegration:
    def test_sparse_embedding_retriever(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=1)

        docs = [
            Document(
                content="Most similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
            ),
            Document(
                content="Less similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[2, 3], values=[0.8, 0.8]),
            ),
        ]
        sparse_document_store.write_documents(docs)

        result = retriever.run(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Most similar sparse document"

    def test_sparse_embedding_retriever_with_filters(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=2)

        docs = [
            Document(
                content="Most similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"type": "match"},
            ),
            Document(
                content="Filtered out sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.95, 0.95]),
                meta={"type": "other"},
            ),
        ]
        sparse_document_store.write_documents(docs)

        result = retriever.run(
            query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]),
            filters={"field": "type", "operator": "==", "value": "match"},
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Most similar sparse document"

    def test_sparse_embedding_retriever_merge_filter_policy(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(
            document_store=sparse_document_store,
            top_k=10,
            filters={"field": "meta.category", "operator": "==", "value": "science"},
            filter_policy="merge",
        )

        docs = [
            Document(
                content="science + en: should match both filters",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "science", "lang": "en"},
            ),
            Document(
                content="science + fr: blocked by runtime filter",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "science", "lang": "fr"},
            ),
            Document(
                content="news + en: blocked by init filter",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "news", "lang": "en"},
            ),
        ]
        sparse_document_store.write_documents(docs)

        result = retriever.run(
            query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]),
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "science + en: should match both filters"

    def test_sparse_embedding_retriever_empty_result(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        # Docs use indices [0, 1]; query uses completely disjoint indices [2, 3]
        sparse_document_store.write_documents(
            [
                Document(
                    content="Sparse doc",
                    sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                )
            ]
        )

        result = retriever.run(query_sparse_embedding=SparseEmbedding(indices=[2, 3], values=[1.0, 1.0]))
        assert result["documents"] == []

    def test_sparse_embedding_retriever_ignores_docs_without_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        docs = [
            Document(content="No sparse embedding"),
            Document(content="Also no sparse embedding", embedding=[0.1, 0.2, 0.3]),
        ]
        sparse_document_store.write_documents(docs)

        # Documents are stored — count_documents sees them
        assert sparse_document_store.count_documents() == 2

        # But sparse retrieval returns nothing — no sparse_vector field to match against
        result = retriever.run(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert result["documents"] == []

    def test_sparse_embedding_retriever_round_trips_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=1)

        # Use out-of-order indices to also verify they are sorted on retrieval
        sparse_document_store.write_documents(
            [
                Document(
                    content="Sparse doc",
                    sparse_embedding=SparseEmbedding(indices=[2, 0, 1], values=[0.5, 0.9, 0.8]),
                )
            ]
        )

        result = retriever.run(query_sparse_embedding=SparseEmbedding(indices=[0, 1, 2], values=[1.0, 1.0, 1.0]))
        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.sparse_embedding is not None
        assert doc.sparse_embedding.indices == [0, 1, 2]
        assert doc.sparse_embedding.values == [0.9, 0.8, 0.5]

    def test_sparse_embedding_retriever_excludes_docs_without_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        docs = [
            Document(
                content="Has sparse embedding",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
            ),
            Document(content="No sparse embedding at all"),
            Document(content="Also no sparse embedding", embedding=[0.1, 0.2, 0.3, 0.4]),
        ]
        sparse_document_store.write_documents(docs)

        result = retriever.run(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Has sparse embedding"

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=1)

        docs = [
            Document(
                content="Most similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
            ),
            Document(
                content="Less similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[2, 3], values=[0.8, 0.8]),
            ),
        ]
        await sparse_document_store.write_documents_async(docs)

        result = await retriever.run_async(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Most similar sparse document"

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_with_filters(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=2)

        docs = [
            Document(
                content="Most similar sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"type": "match"},
            ),
            Document(
                content="Filtered out sparse document",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.95, 0.95]),
                meta={"type": "other"},
            ),
        ]
        await sparse_document_store.write_documents_async(docs)

        result = await retriever.run_async(
            query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]),
            filters={"field": "type", "operator": "==", "value": "match"},
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Most similar sparse document"

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_merge_filter_policy(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(
            document_store=sparse_document_store,
            top_k=10,
            filters={"field": "meta.category", "operator": "==", "value": "science"},
            filter_policy="merge",
        )

        docs = [
            Document(
                content="science + en: should match both filters",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "science", "lang": "en"},
            ),
            Document(
                content="science + fr: blocked by runtime filter",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "science", "lang": "fr"},
            ),
            Document(
                content="news + en: blocked by init filter",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                meta={"category": "news", "lang": "en"},
            ),
        ]
        await sparse_document_store.write_documents_async(docs)

        result = await retriever.run_async(
            query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]),
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "science + en: should match both filters"

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_empty_result(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        # Docs use indices [0, 1]; query uses completely disjoint indices [2, 3]
        await sparse_document_store.write_documents_async(
            [
                Document(
                    content="Sparse doc",
                    sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
                )
            ]
        )

        result = await retriever.run_async(query_sparse_embedding=SparseEmbedding(indices=[2, 3], values=[1.0, 1.0]))
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_ignores_docs_without_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        docs = [
            Document(content="No sparse embedding"),
            Document(content="Also no sparse embedding", embedding=[0.1, 0.2, 0.3]),
        ]
        await sparse_document_store.write_documents_async(docs)

        # Documents are stored — count_documents sees them
        assert await sparse_document_store.count_documents_async() == 2

        # But sparse retrieval returns nothing — no sparse_vector field to match against
        result = await retriever.run_async(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_round_trips_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=1)

        # Use out-of-order indices to also verify they are sorted on retrieval
        await sparse_document_store.write_documents_async(
            [
                Document(
                    content="Sparse doc",
                    sparse_embedding=SparseEmbedding(indices=[2, 0, 1], values=[0.5, 0.9, 0.8]),
                )
            ]
        )

        result = await retriever.run_async(
            query_sparse_embedding=SparseEmbedding(indices=[0, 1, 2], values=[1.0, 1.0, 1.0])
        )
        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.sparse_embedding is not None
        assert doc.sparse_embedding.indices == [0, 1, 2]
        assert doc.sparse_embedding.values == [0.9, 0.8, 0.5]

    @pytest.mark.asyncio
    async def test_sparse_embedding_retriever_async_excludes_docs_without_sparse_embedding(self, sparse_document_store):
        retriever = ElasticsearchSparseEmbeddingRetriever(document_store=sparse_document_store, top_k=10)

        docs = [
            Document(
                content="Has sparse embedding",
                sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.9]),
            ),
            Document(content="No sparse embedding at all"),
            Document(content="Also no sparse embedding", embedding=[0.1, 0.2, 0.3, 0.4]),
        ]
        await sparse_document_store.write_documents_async(docs)

        result = await retriever.run_async(query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0]))
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Has sparse embedding"
