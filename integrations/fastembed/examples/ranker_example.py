from haystack import Document

from haystack_integrations.components.rankers.fastembed import FastembedRanker

query = "Who is maintaining Qdrant?"
documents = [
    Document(
        content="This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc."
    ),
    Document(content="fastembed is supported by and maintained by Qdrant."),
]

ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
ranker.warm_up()
reranked_documents = ranker.run(query=query, documents=documents)["documents"]


print(reranked_documents["documents"][0])

# Document(id=...,
#  content: 'fastembed is supported by and maintained by Qdrant.',
#  score: 5.472434997558594..)
