import os

from haystack import Document
from haystack.preview import Pipeline
from haystack.preview.components.builders.answer_builder import AnswerBuilder
from haystack.preview.components.builders.prompt_builder import PromptBuilder
from haystack.preview.components.generators import GPTGenerator
from astra_store.document_store import AstraDocumentStore
from astra_store.retriever import AstraSingleRetriever, AstraRetriever

# Create a RAG query pipeline
prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """

astra_id = os.getenv("ASTRA_DB_ID", "")
astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

astra_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")
keyspace_name = os.getenv("KEYSPACE_NAME", "recommender_demo")

# We support many different databases. Here, we load a simple and lightweight in-memory database.
document_store = AstraDocumentStore(
    astra_id=astra_id,
    astra_region=astra_region,
    astra_collection=collection_name,
    astra_keyspace=keyspace_name,
    astra_application_token=astra_application_token,
    embedding_dim=384,
)

rag_pipeline = Pipeline()
rag_pipeline.add_component(instance=AstraSingleRetriever(document_store=document_store), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY")), name="llm")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.metadata", "answer_builder.metadata")
rag_pipeline.connect("retriever", "answer_builder.documents")

# Draw the pipeline
rag_pipeline.draw("./rag_pipeline.png")

# Add Documents
documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves."
    ),
]
rag_pipeline.get_component("retriever").document_store.write_documents(documents)

# Run the pipeline
question = "How many languages are there?"
result = rag_pipeline.run(
    {"retriever": {"query": question}, "prompt_builder": {"question": question}, "answer_builder": {"query": question}}
)
print(result["answer_builder"]["answers"][0])