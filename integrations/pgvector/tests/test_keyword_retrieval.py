import pytest
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from numpy.random import rand


@pytest.mark.integration
class TestKeywordRetrieval:
    @pytest.fixture
    def document_store_keyword(self, request):
        connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
        table_name = f"haystack_nn_{request.node.name}"
        embedding_dimension = 768
        vector_function = "cosine_similarity"
        recreate_table = True
        search_strategy = "exact_nearest_neighbor"

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

    @pytest.mark.parametrize("document_store", ["document_store_keyword"], indirect=True)
    def test_keyword_retrieval(self, document_store: PgvectorDocumentStore):
        # Mock query and expected documents
        query = "fox"
        docs = [
            Document(content="The quick brown fox chased the dog", embedding=[0.1] * 768),
            Document(content="The fox was brown", embedding=[0.1] * 768),
            Document(content="The lazy dog", embedding=[0.1] * 768),
        ]

        document_store.write_documents(docs)

        results = document_store._keyword_retrieval(user_query=query, top_k=2)

        assert len(results) == 2
        assert results[0].content == docs[0].content

    @pytest.mark.parametrize("document_store", ["document_store_keyword"], indirect=True)
    def test_keyword_retrieval_with_filters(self, document_store: PgvectorDocumentStore):
        docs = [Document(content=f"Document {i} testing", embedding=rand(768).tolist()) for i in range(10)]

        for i in range(10):
            docs[i].meta["meta_field"] = "custom_value" if i % 2 == 0 else "other_value"

        document_store.write_documents(docs)

        query = "Document"
        filters = {"field": "meta.meta_field", "operator": "==", "value": "custom_value"}

        results = document_store._keyword_retrieval(user_query=query, top_k=3, filters=filters)
        assert len(results) == 3
        assert "meta_field" in results[0].meta
        # for result in results:
        #     assert result.meta["meta_field"] == "custom_value"
