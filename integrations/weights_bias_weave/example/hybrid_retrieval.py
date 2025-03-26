from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.connectors import WeaveConnector


def index():

    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="The Eiffel Tower is located in Paris, France."),
        Document(content="The capital of Germany is Berlin."),
        Document(content="The Colosseum is located in Rome, Italy."),
        Document(content="Paris is the capital of France."),
        Document(content="The Louvre is located in Paris, France."),
        Document(content="The Brandenburg Gate is located in Berlin, Germany."),
    ]

    embedder = SentenceTransformersDocumentEmbedder()
    embedder.warm_up()
    embed_docs = embedder.run(documents)

    document_store.write_documents(embed_docs["documents"])

    return document_store


def hybrid_pipeline(document_store):

    text_embedder = SentenceTransformersTextEmbedder()
    embedding_retriever = InMemoryEmbeddingRetriever(document_store)
    bm25_retriever = InMemoryBM25Retriever(document_store)
    document_joiner = DocumentJoiner()
    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)
    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker")

    connector = WeaveConnector(pipeline_name="test_pipeline")
    hybrid_retrieval.add_component("connector", connector)

    return hybrid_retrieval


def main():
    doc_store = index()
    pipeline = hybrid_pipeline(doc_store)

    query = "What is the capital of Germany?"
    r = pipeline.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}})
    print(r)  # noqa: T201


if __name__ == "__main__":
    main()
