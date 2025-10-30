from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.rankers.fastembed import FastembedColbertReranker

store = InMemoryDocumentStore()
store.write_documents([
    Document(content="ColBERT uses late interaction over token embeddings."),
    Document(content="BM25 is a classic sparse retrieval model."),
    Document(content="Cross-encoders directly score query–document pairs."),
])

retriever = InMemoryBM25Retriever(document_store=store)
reranker = FastembedColbertReranker(model="colbert-ir/colbertv2.0")

query = "late interaction reranking"
cand = retriever.run(query=query, top_k=200)
out = reranker.run(query=query, documents=cand["documents"], top_k=5)
for i, d in enumerate(out["documents"], 1):
    print(i, round(d.score, 3), d.content)
