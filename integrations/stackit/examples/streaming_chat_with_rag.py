# To run this example, you will need to set a `STACKIT_API_KEY` environment variable.
# The HTMLToDocument requires an additional dependency to be installed via `pip install trafilatura`.
# This example streams chat replies to the console.

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.generators.stackit import STACKITChatGenerator

document_store = InMemoryDocumentStore()
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
chunker = DocumentSplitter()
writer = DocumentWriter(document_store=document_store)

indexing = Pipeline()

indexing.add_component(name="fetcher", instance=fetcher)
indexing.add_component(name="converter", instance=converter)
indexing.add_component(name="chunker", instance=chunker)
indexing.add_component(name="writer", instance=writer)

indexing.connect("fetcher", "converter")
indexing.connect("converter", "chunker")
indexing.connect("chunker", "writer")

indexing.run(data={"fetcher": {"urls": ["https://www.stackit.de/en/"]}})

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(variables=["documents"])
llm = STACKITChatGenerator(
    model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8", streaming_callback=print_streaming_chunk
)

messages = [ChatMessage.from_user("Here are some of the documents: {{documents}} \\n Question: {{query}} \\n Answer:")]

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What does STACKIT offer?"

result = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"template_variables": {"query": question}, "template": messages},
        "llm": {"generation_kwargs": {"max_tokens": 165}},
    }
)

print(result)  # noqa: T201
