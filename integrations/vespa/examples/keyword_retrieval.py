# Before running this example, make sure you have a Vespa application running and
# a schema configured with fields compatible with the document store arguments below.
#
# Install the required packages with:
# pip install vespa-haystack "sentence-transformers>=2.2.0"
#
# Set the `VESPA_URL` environment variable to your Vespa endpoint, for example:
# export VESPA_URL="http://localhost"

import logging

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.vespa import VespaKeywordRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore

logger = logging.getLogger(__name__)

# This example assumes your Vespa schema already exists and contains:
# - a text field named `content`
# - metadata fields `category` and `author`
# - a ranking profile named `bm25`
document_store = VespaDocumentStore(
    schema="doc",
    namespace="doc",
    content_field="content",
    metadata_fields=["category", "author"],
)

indexing = Pipeline()
indexing.add_component("writer", DocumentWriter(document_store=document_store))

indexing.run(
    {
        "writer": {
            "documents": [
                Document(id="1", content="Haystack integrates with Vespa for search.", meta={"category": "docs"}),
                Document(id="2", content="Vespa supports lexical and vector retrieval.", meta={"category": "docs"}),
                Document(id="3", content="This note is about something else entirely.", meta={"category": "misc"}),
            ]
        }
    }
)

querying = Pipeline()
querying.add_component(
    "retriever",
    VespaKeywordRetriever(
        document_store=document_store,
        top_k=2,
        filters={"field": "meta.category", "operator": "==", "value": "docs"},
    ),
)

results = querying.run({"retriever": {"query": "vector retrieval"}})

for doc in results["retriever"]["documents"]:
    logger.info("%s", doc)
