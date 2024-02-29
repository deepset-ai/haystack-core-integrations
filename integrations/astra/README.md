[![test](https://github.com/deepset-ai/document-store/actions/workflows/test.yml/badge.svg)](https://github.com/deepset-ai/document-store/actions/workflows/test.yml)

# Astra Store

## Installation

```bash
pip install astra-haystack

```

### Local Development
install astra-haystack package locally to run integration tests:

Open in gitpod:
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/Anant/astra-haystack/tree/main)

Switch Python version to 3.9 (Requires 3.8+ but not 3.12)
```
pyenv install 3.9
pyenv local 3.9
```

Local install for the package
`pip install -e .`
To execute integration tests, add needed environment variables
`ASTRA_DB_API_ENDPOINT=<id>`
`ASTRA_DB_APPLICATION_TOKEN=<token>`
and execute
`python examples/example.py`

Install requirements
`pip install -r requirements.txt`

Export environment variables
```
export ASTRA_DB_API_ENDPOINT=
export ASTRA_DB_APPLICATION_TOKEN=
export COLLECTION_NAME=
export OPENAI_API_KEY=
```

run the python examples
`python example/example.py`
or
`python example/pipeline_example.py`

## Usage

This package includes Astra Document Store and Astra Embedding Retriever classes that integrate with Haystack, allowing you to easily perform document retrieval or RAG with Astra, and include those functions in Haystack pipelines.

### In order to use the Document Store directly:

Import the Document Store:
```
from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack.document_stores.types.policy import DuplicatePolicy
```

Load in environment variables:
```
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT", "")
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")
```

Create the Document Store object:
```
document_store = AstraDocumentStore(
    api_endpoint=api_endpoint,
    token=token,
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)
```

Then you can use the document store functions like count_document below:
`document_store.count_documents()`

### Using the Astra Embedding Retriever with Haystack Pipelines

Create the Document Store object like above, then import and create the Pipeline:

```
from haystack import Pipeline
pipeline = Pipeline()
```

Add your AstraEmbeddingRetriever into the pipeline
`pipeline.add_component(instance=AstraEmbeddingRetriever(document_store=document_store), name="retriever")`

Add other components and connect them as desired. Then run your pipeline:
`pipeline.run(...)`
