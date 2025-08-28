from typing import List

import pytest
from haystack.dataclasses import Document

from haystack_integrations.document_stores.opensearch.document_store import OpenSearchDocumentStore


@pytest.mark.integration
class TestDocumentStoreAsync:
    @pytest.mark.asyncio
    async def test_write_documents(self, document_store: OpenSearchDocumentStore):
        assert await document_store.write_documents_async([Document(id="1")]) == 1

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore, test_documents: List[Document]):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional", top_k=3)

        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_pagination(
        self, document_store: OpenSearchDocumentStore, test_documents: List[Document]
    ):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("programming", top_k=11)

        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match(
        self, document_store: OpenSearchDocumentStore, test_documents: List[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=3, all_terms_must_match=True)

        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match_false(
        self, document_store: OpenSearchDocumentStore, test_documents: List[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=10, all_terms_must_match=False)

        assert len(res) == 5
        assert all("functional" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_filters(
        self, document_store: OpenSearchDocumentStore, test_documents: List[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async(
            "programming",
            top_k=10,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )

        assert len(res) == 5
        retrieved_ids = sorted([doc.id for doc in res])
        assert retrieved_ids == ["1", "2", "3", "4", "5"]

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_custom_query(
        self, document_store: OpenSearchDocumentStore, test_documents: List[Document]
    ):
        document_store.write_documents(test_documents)

        custom_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": {"match": {"content": "$query"}},
                            "filter": "$filters",
                        }
                    },
                    "field_value_factor": {
                        "field": "likes",
                        "factor": 0.1,
                        "modifier": "log1p",
                        "missing": 0,
                    },
                }
            }
        }
        res = await document_store._bm25_retrieval_async(
            "functional",
            top_k=3,
            custom_query=custom_query,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    @pytest.mark.asyncio
    async def test_embedding_retrieval(self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_filters(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_custom_query(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 3}}}],
                    "filter": "$filters",
                }
            }
        }

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            top_k=1,
            filters=filters,
            custom_query=custom_query,
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_but_dont_return_embeddings_for_embedding_retrieval(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_count_documents(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_filter_documents(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(
                content="1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content="2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )

        assert len(result) == 1
        assert result[0].content == "2"
        assert result[0].meta["number"] == 100

    @pytest.mark.asyncio
    async def test_delete_documents(self, document_store: OpenSearchDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert document_store.count_documents() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0
