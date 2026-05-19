# Before running this example, make sure you have a Vespa application running and
# a schema configured with:
# - a text field named `content`
# - a tensor field named `embedding`
# - a ranking profile named `semantic` that supports nearest-neighbor search
#
# Install the required packages with:
# pip install vespa-haystack "sentence-transformers>=2.2.0"
#
# Set the `VESPA_URL` environment variable to your Vespa endpoint, for example:
# export VESPA_URL="http://localhost"

import logging

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.vespa import VespaEmbeddingRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore

logger = logging.getLogger(__name__)

document_store = VespaDocumentStore(
    schema="doc",
    namespace="doc",
    content_field="content",
    embedding_field="embedding",
    metadata_fields=["category"],
)

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store=document_store))
indexing.connect("embedder", "writer")

indexing.run(
    {
        "embedder": {
            "documents": [
                Document(id="1", content="Haystack integrates with Vespa for search.", meta={"category": "docs"}),
                Document(id="2", content="Vespa supports lexical and vector retrieval.", meta={"category": "docs"}),
                Document(id="3", content="Cats sleep most of the day.", meta={"category": "animals"}),
            ]
        }
    }
)

querying = Pipeline()
querying.add_component("text_embedder", SentenceTransformersTextEmbedder())
querying.add_component(
    "retriever",
    VespaEmbeddingRetriever(
        document_store=document_store,
        top_k=2,
        query_tensor_name="query_embedding",
    ),
)
querying.connect("text_embedder", "retriever")

results = querying.run({"text_embedder": {"text": "semantic vector search"}})

for doc in results["retriever"]["documents"]:
    logger.info("%s", doc)
