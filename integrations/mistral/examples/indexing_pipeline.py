# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# To run this example, set the `MISTRAL_API_KEY` environment variable
# and install `trafilatura`, used by HTMLToDocument: `pip install trafilatura`

from haystack import Pipeline
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder

document_store = InMemoryDocumentStore()
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
chunker = DocumentSplitter()
embedder = MistralDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)

indexing = Pipeline()

indexing.add_component(name="fetcher", instance=fetcher)
indexing.add_component(name="converter", instance=converter)
indexing.add_component(name="chunker", instance=chunker)
indexing.add_component(name="embedder", instance=embedder)
indexing.add_component(name="writer", instance=writer)

indexing.connect("fetcher", "converter")
indexing.connect("converter", "chunker")
indexing.connect("chunker", "embedder")
indexing.connect("embedder", "writer")

indexing.run(data={"fetcher": {"urls": ["https://mistral.ai/news/la-plateforme/"]}})
