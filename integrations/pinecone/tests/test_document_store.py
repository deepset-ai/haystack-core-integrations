import time

from haystack import Document

from pinecone_haystack.document_store import PineconeDocumentStore


class TestDocumentStore:
    def test_embedding_retrieval(self, document_store: PineconeDocumentStore, sleep_time):
        docs = [
            Document(content="Most similar document", embedding=[1.0] * 10),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5]),
            Document(content="Not very similar document", embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
        document_store.write_documents(docs)
        time.sleep(sleep_time)
        results = document_store._embedding_retrieval(query_embedding=[0.1] * 10, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"
