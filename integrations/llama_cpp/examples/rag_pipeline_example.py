from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores import InMemoryDocumentStore

from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator

# Load first 100 rows of the Simple Wikipedia Dataset from HuggingFace
dataset = load_dataset("pszemraj/simple_wikipedia", split="validation[:100]")

docs = [
    Document(
        content=doc["text"],
        meta={
            "title": doc["title"],
            "url": doc["url"],
        },
    )
    for doc in dataset
]

doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=doc_embedder, name="doc_embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=doc_store), name="doc_writer")
indexing_pipeline.connect("doc_embedder", "doc_writer")

indexing_pipeline.run({"doc_embedder": {"documents": docs}})


# RAG Pipeline
template = [
    ChatMessage.from_user(
        """Answer the question using the provided context.
Question: {{question}}
Context:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Answer:"""
    )
]

rag_pipeline = Pipeline()

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

model_path = "openchat-3.5-1210.Q3_K_S.gguf"
generator = LlamaCppChatGenerator(model=model_path, n_ctx=4096, n_batch=128)

rag_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
rag_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=doc_store, top_k=3), name="retriever")
rag_pipeline.add_component(
    instance=ChatPromptBuilder(template=template, variables=["question", "documents"]), name="prompt_builder"
)
rag_pipeline.add_component(instance=generator, name="llm")

rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


# Run Pipeline
question = "Which year did the Joker movie release?"
result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "llm": {"generation_kwargs": {"max_tokens": 128, "temperature": 0.1}},
    }
)

print(result["llm"]["replies"][0].text)
# The Joker movie was released on October 4, 2019.
