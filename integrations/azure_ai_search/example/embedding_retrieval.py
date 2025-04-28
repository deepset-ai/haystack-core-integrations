from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.azure_ai_search import AzureAISearchEmbeddingRetriever
from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

"""
This example demonstrates how to use the AzureAISearchEmbeddingRetriever to retrieve documents
using embeddings based on a query. To run this example, you'll need an Azure Search service endpoint
and API key, which can either be
set as environment variables (AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY) or
provided directly to AzureAISearchDocumentStore(as params "api_key", "azure_endpoint").
Otherwise you can use DefaultAzureCredential to authenticate with Azure services.
See more details at https://learn.microsoft.com/en-us/azure/search/keyless-connections?tabs=python%2Cazure-cli
"""

document_store = AzureAISearchDocumentStore(index_name="retrieval-example")

model = "sentence-transformers/all-mpnet-base-v2"

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="""Elephants have been observed to behave in a way that indicates a
         high level of self-awareness, such as recognizing themselves in mirrors."""
    ),
    Document(
        content="""In certain parts of the world, like the Maldives, Puerto Rico, and
          San Diego, you can witness the phenomenon of bioluminescent waves."""
    ),
]

document_embedder = SentenceTransformersDocumentEmbedder(model=model)
document_embedder.warm_up()

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=document_embedder, name="doc_embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="doc_writer")
indexing_pipeline.connect("doc_embedder", "doc_writer")

indexing_pipeline.run({"doc_embedder": {"documents": documents}})

# Query Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model))
query_pipeline.add_component("retriever", AzureAISearchEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result["retriever"]["documents"][0])
