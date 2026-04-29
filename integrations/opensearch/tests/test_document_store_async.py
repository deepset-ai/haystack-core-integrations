# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)

from haystack_integrations.document_stores.opensearch.document_store import OpenSearchDocumentStore
from tests.test_document_store_common import OpenSearchDocumentStoreTestMixin


@pytest.mark.integration
class TestDocumentStoreAsync(
    OpenSearchDocumentStoreTestMixin,
    WriteDocumentsAsyncTest,
    CountDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    UpdateByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, DuplicatePolicy.FAIL)

    @pytest.mark.asyncio
    async def test_count_not_empty_async(self, document_store: OpenSearchDocumentStore):
        # Override: haystack v2.28.0 is missing @staticmethod on this mixin method.
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore, test_documents: list[Document]):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional", top_k=3)

        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_pagination(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
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
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=3, all_terms_must_match=True)

        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match_false(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=10, all_terms_must_match=False)

        assert len(res) == 5
        assert all("functional" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_fuzziness(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)

        query_with_typo = "functinal"
        # Query without fuzziness to search for the exact match
        res = await document_store._bm25_retrieval_async(query_with_typo, top_k=3, fuzziness="0")
        # Nothing is found as the query contains a typo
        assert res == []

        # Query with fuzziness with the same query
        res = await document_store._bm25_retrieval_async(query_with_typo, top_k=3, fuzziness="1")
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_fuzziness_overflow(self, document_store: OpenSearchDocumentStore, caplog):
        """
        Test that a long query with fuzziness="AUTO" that exceeds OpenSearch's maxClauseCount
        is automatically retried with fuzziness=0 instead of raising an error.
        """
        # Build an index vocabulary of similar 5-character words. With fuzziness="AUTO",
        # 5-char words get edit distance 1, so each query term fuzzy-matches many similar
        # indexed terms, causing clause expansion beyond the default maxClauseCount (1024).
        # With fuzziness=0, each term produces exactly 1 clause, staying well under the limit.
        words = [f"foo{chr(97 + i)}{chr(97 + j)}" for i in range(20) for j in range(26)]  # 520 words

        chunk_size = 52
        docs = [
            Document(content=" ".join(words[i : i + chunk_size]), id=str(idx))
            for idx, i in enumerate(range(0, len(words), chunk_size))
        ]
        document_store.write_documents(docs)

        # Query with a subset of words. With fuzziness="AUTO", each 5-char term expands
        # to match ~45 similar indexed terms, pushing total clauses well above 1024.
        long_query = " ".join(words[:100])

        # This should not raise: the too_many_clauses error is caught and retried with fuzziness=0
        res = await document_store._bm25_retrieval_async(long_query, top_k=3, fuzziness="AUTO")
        assert isinstance(res, list)
        assert "Retrying with fuzziness=0" in caplog.text

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_filters(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
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
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
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
    async def test_write_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test async writing documents with routing metadata"""
        docs = [
            Document(id="1", content="User A doc", meta={"_routing": "user_a", "category": "test"}),
            Document(id="2", content="User B doc", meta={"_routing": "user_b"}),
            Document(id="3", content="No routing"),
        ]

        written = await document_store.write_documents_async(docs)
        assert written == 3
        assert await document_store.count_documents_async() == 3

        # Verify _routing not stored in metadata
        retrieved = await document_store.filter_documents_async()
        retrieved_by_id = {doc.id: doc for doc in retrieved}

        # Check _routing is not stored for any document
        for doc in retrieved:
            assert "_routing" not in doc.meta

        assert retrieved_by_id["1"].meta["category"] == "test"
        assert retrieved_by_id["2"].meta == {}
        assert retrieved_by_id["3"].meta == {}

    @pytest.mark.asyncio
    async def test_delete_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test async deleting documents with routing"""
        docs = [
            Document(id="1", content="Doc 1", meta={"_routing": "user_a"}),
            Document(id="2", content="Doc 2", meta={"_routing": "user_b"}),
            Document(id="3", content="Doc 3"),
        ]
        await document_store.write_documents_async(docs)

        routing_map = {"1": "user_a", "2": "user_b"}
        await document_store.delete_documents_async(["1", "2"], routing=routing_map)
        assert await document_store.count_documents_async() == 1

    @pytest.mark.asyncio
    async def test_metadata_search_async_fuzzy_mode(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search in fuzzy mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 documents with category "Python"
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)

    @pytest.mark.asyncio
    async def test_metadata_search_async_strict_mode(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search in strict mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="strict",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) >= 1  # At least 1 document with category "Python"
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)

    @pytest.mark.asyncio
    async def test_metadata_search_async_multiple_fields(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search across multiple fields."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="active",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(row, dict) for row in result)
        # Results should only contain the specified fields
        for row in result:
            assert all(key in ["category", "status"] for key in row.keys())

    @pytest.mark.asyncio
    async def test_metadata_search_async_top_k(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search respects top_k parameter."""
        docs = [Document(content=f"Doc {i}", meta={"category": "Python", "index": i}) for i in range(15)]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=5,
        )

        assert isinstance(result, list)
        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_metadata_search_async_comma_separated_query(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with comma-separated query parts."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python, active",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(row, dict) for row in result)

    @pytest.mark.asyncio
    async def test_metadata_search_async_with_filters(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with additional filters."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Python", "status": "inactive", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Java", "status": "active", "priority": 1}),
        ]
        document_store.write_documents(docs, refresh=True)

        filters = {"field": "priority", "operator": "==", "value": 1}
        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=10,
            filters=filters,
        )

        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_metadata_search_async_empty_fields(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with empty fields list returns empty result."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=[],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_metadata_search_async_deduplication(self, document_store: OpenSearchDocumentStore):
        """Test that async metadata search deduplicates results."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active"}),
            Document(content="Doc 2", meta={"category": "Python", "status": "active"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        seen = []
        for row in result:
            row_tuple = tuple(sorted(row.items()))
            assert row_tuple not in seen, "Duplicate metadata found"
            seen.append(row_tuple)

    @pytest.mark.asyncio
    async def test_query_sql(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
            Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
        ]
        await document_store.write_documents_async(docs, refresh=True)

        sql_query = (
            f"SELECT content, category, status, priority FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY priority"
        )
        result = await document_store._query_sql_async(sql_query)

        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "status", "priority"]
        assert len(result["datarows"]) == 2  # Two documents with category A

        categories = [row[1] for row in result["datarows"]]
        assert all(category == "A" for category in categories)

        invalid_query = "SELECT * FROM non_existent_index"
        with pytest.raises(DocumentStoreError, match="Failed to execute SQL query"):
            await document_store._query_sql_async(invalid_query)

    @pytest.mark.asyncio
    async def test_query_sql_async_with_fetch_size(self, document_store: OpenSearchDocumentStore):
        """Test async SQL query with fetch_size parameter"""
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
        await document_store.write_documents_async(docs, refresh=True)

        sql_query = (
            f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY index"
        )

        result = await document_store._query_sql_async(sql_query, fetch_size=5)

        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "index"]
        assert len(result["datarows"]) > 0
        assert len(result["datarows"]) <= 5
        assert result.get("cursor") is not None

    @pytest.mark.asyncio
    async def test_explicit_nested_fields_filter(self, document_store_nested: OpenSearchDocumentStore):
        """Filtering on explicitly declared nested fields returns correct documents (async)."""
        docs = [
            Document(
                content="doc about bgb 1a",
                meta={"refs": [{"law": "bgb", "section": "1", "paragraph": "a"}], "status": "active"},
            ),
            Document(
                content="doc about bgb 2",
                meta={"refs": [{"law": "bgb", "section": "2"}], "status": "active"},
            ),
            Document(
                content="doc about stgb",
                meta={"refs": [{"law": "stgb", "section": "1"}], "status": "active"},
            ),
        ]
        await document_store_nested.write_documents_async(docs)

        results = await document_store_nested.filter_documents_async(
            filters={"field": "meta.refs.law", "operator": "==", "value": "bgb"}
        )
        assert len(results) == 2
        assert all("bgb" in str(doc.meta["refs"]) for doc in results)

    @pytest.mark.asyncio
    async def test_explicit_nested_fields_combined_filter(self, document_store_nested: OpenSearchDocumentStore):
        """AND filter across sub-fields of the same nested path matches within the same array element (async)."""
        docs = [
            Document(
                content="bgb section 1",
                meta={"refs": [{"law": "bgb", "section": "1"}, {"law": "stgb", "section": "2"}]},
            ),
            Document(
                content="bgb section 2",
                meta={"refs": [{"law": "bgb", "section": "2"}]},
            ),
            Document(
                content="stgb section 1",
                meta={"refs": [{"law": "stgb", "section": "1"}]},
            ),
        ]
        await document_store_nested.write_documents_async(docs)

        results = await document_store_nested.filter_documents_async(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.refs.section", "operator": "==", "value": "1"},
                ],
            }
        )
        assert len(results) == 1
        assert results[0].content == "bgb section 1"

    @pytest.mark.asyncio
    async def test_query_sql_async_pagination_flow(self, document_store: OpenSearchDocumentStore):
        """Test async pagination flow with fetch_size"""
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(20)]
        await document_store.write_documents_async(docs, refresh=True)

        sql_query = (
            f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY index"
        )

        result = await document_store._query_sql_async(sql_query, fetch_size=10)
        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "index"]
        assert len(result["datarows"]) > 0
        assert len(result["datarows"]) <= 10

        for row in result["datarows"]:
            assert len(row) == 3
