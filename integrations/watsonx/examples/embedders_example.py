# In order to run this example, you will need watsonx credentials.
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.watsonx.document_embedder import WatsonxDocumentEmbedder
from haystack_integrations.components.embedders.watsonx.text_embedder import WatsonxTextEmbedder

# Step 1: Set your credentials as env var

# Step 2: Create document store
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Step 3: Prepare documents
documents = [
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="My name is Wolfgang and I live in Berlin"),
]

# Step 4: Embed documents
document_embedder = WatsonxDocumentEmbedder(
    model="ibm/slate-125m-english-rtrvr",
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    api_base_url="https://us-south.ml.cloud.ibm.com",
)
documents_with_embeddings = document_embedder.run(documents)["documents"]

document_store.write_documents(documents_with_embeddings)

# Step 5: Build pipeline
query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder",
    WatsonxTextEmbedder(
        model="ibm/slate-125m-english-rtrvr",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        api_base_url="https://us-south.ml.cloud.ibm.com",
    ),
)
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# Step 6: Run query
query = "Who lives in Berlin?"
result = query_pipeline.run({"text_embedder": {"text": query}})

# Step 7: Print result
doc = result["retriever"]["documents"][0]
print("\nTop Result:")
print(f"Content: {doc.content}")
print(f"Score: {doc.score}")
### Expected Result
# Top Result:
# Content: My name is Wolfgang and I live in Berlin
# Score: 0.7287276769333577
