# To run this example, you will need an to set a `MISTRAL_API_KEY` environment variable.
# This example streams chat replies to the console.

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator

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

text_embedder = MistralTextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(variables=["documents"])
llm = MistralChatGenerator(streaming_callback=print_streaming_chunk)

messages = [ChatMessage.from_user("Here are some the documents: {{documents}} \\n Answer: {{query}}")]

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)


rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What are the available models?"

result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"template_variables": {"query": question}, "template": messages},
        "llm": {"generation_kwargs": {"max_tokens": 165}},
    }
)
