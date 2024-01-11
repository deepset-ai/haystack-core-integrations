[![test](https://github.com/deepset-ai/document-store/actions/workflows/test.yml/badge.svg)](https://github.com/deepset-ai/document-store/actions/workflows/test.yml)

# Astra Store

## Installation
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
`ASTRA_DB_ID=<id>`
`ASTRA_DB_APPLICATION_TOKEN=<token>`
and execute
`python examples/example.py`

Install requirements
`pip install -r requirements.txt`

Export environment variables
```
export KEYSPACE_NAME=
export COLLECTION_NAME=
export OPENAI_API_KEY=
export ASTRA_DB_ID=
export ASTRA_DB_REGION=
export ASTRA_DB_APPLICATION_TOKEN=
```

run the python examples
`python example/example.py`
or
`python example/pipeline_example.py`

## Usage

This package includes Astra Document Store and Astra Retriever classes that integrate with Haystack, allowing you to easily perform document retrieval or RAG with Astra, and include those functions in Haystack pipelines.

### In order to use the Document Store directly:

Import the Document Store:
```
from astra_store.document_store import AstraDocumentStore
from haystack.preview.document_stores import DuplicatePolicy
```

Load in environment variables:
```
astra_id = os.getenv("ASTRA_DB_ID", "")
astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

astra_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")
keyspace_name = os.getenv("KEYSPACE_NAME", "recommender_demo")
```

Create the Document Store object:
```
document_store = AstraDocumentStore(
    astra_id=astra_id,
    astra_region=astra_region,
    astra_collection=collection_name,
    astra_keyspace=keyspace_name,
    astra_application_token=astra_application_token,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)
```

Then you can use the document store functions like count_document below:
`document_store.count_documents()`

### Using the Astra Retriever with Haystack Pipelines

Create the Document Store object like above, then import and create the Pipeline:

```
from haystack.preview import Pipeline
pipeline = Pipeline()
```

Add your AstraRetriever into the pipeline
`pipeline.add_component(instance=AstraSingleRetriever(document_store=document_store), name="retriever")`

Add other components and connect them as desired. Then run your pipeline:
`pipeline.run(...)`
