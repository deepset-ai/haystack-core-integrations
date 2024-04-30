import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.connectors.langfuse import LangfuseConnector


def get_pipeline(document_store: InMemoryDocumentStore):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=2)

    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("tracer", LangfuseConnector("Basic RAG Pipeline"))
    basic_rag_pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo", generation_kwargs={"n": 2}))

    # Now, connect the components to each other
    # NOTE: the tracer component doesn't need to be connected to anything in order to work
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    return basic_rag_pipeline


if __name__ == "__main__":
    document_store = InMemoryDocumentStore()
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    embedder = SentenceTransformersDocumentEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    embedder.warm_up()
    docs_with_embeddings = embedder.run([Document(**ds) for ds in dataset]).get("documents") or []  # type: ignore
    document_store.write_documents(docs_with_embeddings)

    pipeline = get_pipeline(document_store)
    question = "What does Rhodes Statue look like?"
    response = pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

    print(response["llm"]["replies"][0])
    print(response["tracer"]["trace_url"])
