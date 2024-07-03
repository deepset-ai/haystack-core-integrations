from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.rankers.cohere import CohereRanker

# Note set your API key by running the below command in your terminal
# export CO_API_KEY="<your Cohere API key>"

docs = [
    Document(content="Paris is in France"),
    Document(content="Berlin is in Germany"),
    Document(content="Lyon is in France"),
]
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)
ranker = CohereRanker(model="rerank-english-v2.0")

document_ranker_pipeline = Pipeline()
document_ranker_pipeline.add_component(instance=retriever, name="retriever")
document_ranker_pipeline.add_component(instance=ranker, name="ranker")

document_ranker_pipeline.connect("retriever.documents", "ranker.documents")

query = "Cities in France"
res = document_ranker_pipeline.run(data={"retriever": {"query": query}, "ranker": {"query": query, "top_k": 2}})
print(res["ranker"]["documents"])  # noqa: T201
