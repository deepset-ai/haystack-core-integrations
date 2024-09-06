import pytest
from haystack.dataclasses.document import Document

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
class TestKeywordRetrieval:
    def test_keyword_retrieval(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(content="The quick brown fox chased the dog", embedding=[0.1] * 768),
            Document(content="The fox was brown", embedding=[0.1] * 768),
            Document(content="The lazy dog", embedding=[0.1] * 768),
            Document(content="fox fox fox", embedding=[0.1] * 768),
        ]

        document_store.write_documents(docs)

        results = document_store._keyword_retrieval(query="fox", top_k=2)

        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
        assert results[0].id == docs[-1].id
        assert results[0].score > results[1].score

    def test_keyword_retrieval_with_filters(self, document_store: PgvectorDocumentStore):
        docs = [
            Document(
                content="The quick brown fox chased the dog",
                embedding=([0.1] * 768),
                meta={"meta_field": "right_value"},
            ),
            Document(content="The fox was brown", embedding=([0.1] * 768), meta={"meta_field": "right_value"}),
            Document(content="The lazy dog", embedding=([0.1] * 768), meta={"meta_field": "right_value"}),
            Document(content="fox fox fox", embedding=([0.1] * 768), meta={"meta_field": "wrong_value"}),
        ]

        document_store.write_documents(docs)

        filters = {"field": "meta.meta_field", "operator": "==", "value": "right_value"}

        results = document_store._keyword_retrieval(query="fox", top_k=3, filters=filters)
        assert len(results) == 2
        for doc in results:
            assert "fox" in doc.content
            assert doc.meta["meta_field"] == "right_value"

    def test_empty_query(self, document_store: PgvectorDocumentStore):
        with pytest.raises(ValueError):
            document_store._keyword_retrieval(query="")
