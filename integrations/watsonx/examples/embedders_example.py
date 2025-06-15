# In order to run this example, you will need watsonx credentials.
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.watsonx.document_embedder import WatsonXDocumentEmbedder
from haystack_integrations.components.embedders.watsonx.text_embedder import WatsonXTextEmbedder

# Step 1: Set your credentials
api_key = '<api-key>'
project_id = '<project-id>'

# Step 2: Create document store
document_store = InMemoryDocumentStore(embedding_similarity_function='cosine')

# Step 3: Prepare documents
documents = [
    Document(content='I saw a black horse running'),
    Document(content='Germany has many big cities'),
    Document(content='My name is Wolfgang and I live in Berlin'),
]

# Step 4: Embed documents
document_embedder = WatsonXDocumentEmbedder(
    model='ibm/slate-125m-english-rtrvr',
    api_key=Secret.from_token(api_key),
    project_id=project_id,
    url='https://us-south.ml.cloud.ibm.com',
)
documents_with_embeddings = document_embedder.run(documents)['documents']

document_store.write_documents(documents_with_embeddings)

# Step 5: Build pipeline
query_pipeline = Pipeline()
query_pipeline.add_component(
    'text_embedder',
    WatsonXTextEmbedder(
        model='ibm/slate-125m-english-rtrvr',
        api_key=Secret.from_token(api_key),
        project_id=project_id,
        url='https://us-south.ml.cloud.ibm.com',
    ),
)
query_pipeline.add_component('retriever', InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')

# Step 6: Run query
query = 'Who lives in Berlin?'
result = query_pipeline.run({'text_embedder': {'text': query}})

# Step 7: Print result

### Result
# Document(id=62fad790ad2af927af9432c87330ed2ea5e31332cdec8e9d6235a5105ab0aaf5, content: 'My name is Wolfgang and I live in Berlin', score: 0.7287276769333577)
