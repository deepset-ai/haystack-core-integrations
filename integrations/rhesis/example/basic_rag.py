import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.connectors.rhesis import RhesisConnector

if __name__ == "__main__":
    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="Berlin is the capital and largest city of Germany."),
        Document(content="Paris is the capital of France."),
    ]
    indexing = Pipeline()
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.run({"writer": {"documents": documents}})

    rag = Pipeline()
    rag.add_component("tracer", RhesisConnector("Basic RAG example"))
    rag.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    rag.add_component("prompt_builder", ChatPromptBuilder())
    rag.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    rag.connect("retriever.documents", "prompt_builder.documents")
    rag.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Answer using only the provided documents."),
        ChatMessage.from_user("{{ query }}"),
    ]

    response = rag.run(
        data={
            "retriever": {"query": "What is the capital of Germany?"},
            "prompt_builder": {"template": messages},
            "tracer": {"invocation_context": {"session_id": "rag-example"}},
        }
    )
    print(response["llm"]["replies"][0])
    print(response["tracer"]["trace_url"])
    print(response["tracer"]["trace_id"])
