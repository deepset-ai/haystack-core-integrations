import os

from haystack import Document, Pipeline, logging
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create a RAG query pipeline
prompt_template = """
Given these documents, answer the question.

Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Question: {{question}}

Answer:
"""

collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")

# Make sure ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN environment variables are set before proceeding

# We support many different databases. Here, we load a simple and lightweight in-memory database.
document_store = AstraDocumentStore(
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dimension=384,
)


# Add Documents
documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="Elephants have been observed to behave in a way that indicates"
        " a high level of self-awareness, such as recognizing themselves in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, "
        "and San Diego, you can witness the phenomenon of bioluminescent waves."
    ),
]
p = Pipeline()
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    name="embedder",
)
p.add_component(instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP), name="writer")
p.connect("embedder.documents", "writer.documents")

p.run({"embedder": {"documents": documents}})


# Construct rag pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component(
    instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    name="embedder",
)
rag_pipeline.add_component(instance=AstraEmbeddingRetriever(document_store=document_store), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
rag_pipeline.connect("embedder", "retriever")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("retriever", "answer_builder.documents")


# Draw the pipeline
rag_pipeline.draw("./rag_pipeline.png")


# Run the pipeline
question = "How many languages are there in the world today?"
result = rag_pipeline.run(
    {
        "embedder": {"text": question},
        "retriever": {"top_k": 2},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)

logger.info(result)
