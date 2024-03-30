# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class TestKeywordRetrieval:
    @pytest.fixture
    def document_store_w_hnsw_index(self, request):
        connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
        table_name = f"haystack_hnsw_{request.node.name}"
        embedding_dimension = 768
        vector_function = "cosine_similarity"
        recreate_table = True
        search_strategy = "hnsw"

        store = PgvectorDocumentStore(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dimension=embedding_dimension,
            vector_function=vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )
        yield store

        store.delete_table()

    @pytest.mark.parametrize("document_store", ["document_store", "document_store_w_hnsw_index"], indirect=True)
    def test_keyword_retrieval(self, document_store: PgvectorDocumentStore):
        query = "Most similar/best keyword document (cosine sim)"
        docs = [
            Document(content="Most similar keyword document (cosine sim)"),
            Document(content="2nd best keyword document (cosine sim)"),
            Document(content="Not very similar document (cosine sim)"),
        ]

        document_store.write_documents(docs)

        results = document_store._keyword_retrieval(user_query=query, top_k=2, filters={}, language="english")
        assert len(results) == 2
        assert results[0].content == "Most similar document (cosine sim)"
        assert results[1].content == "2nd best document (cosine sim)"
        assert results[0].score > results[1].score
