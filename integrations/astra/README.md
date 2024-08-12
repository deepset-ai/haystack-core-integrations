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
`ASTRA_DB_API_ENDPOINT="https://<id>-<region>.apps.astra.datastax.com"`,
`ASTRA_DB_APPLICATION_TOKEN="AstraCS:..."`
and execute
`python examples/example.py`

Install requirements
`pip install -r requirements.txt`

Export environment variables
```
export ASTRA_DB_API_ENDPOINT="https://<id>-<region>.apps.astra.datastax.com"
export ASTRA_DB_APPLICATION_TOKEN="AstraCS:..."
export COLLECTION_NAME="my_collection"
export OPENAI_API_KEY="sk-..."
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
namespace = os.environ.get("ASTRA_DB_KEYSPACE")
collection_name = os.environ.get("COLLECTION_NAME", "haystack_vector_search")
```

Create the Document Store object (API Endpoint and Token are read off the environment):
```
document_store = AstraDocumentStore(
    collection_name=collection_name,
    namespace=namespace,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dimension=384,
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

## Warnings about indexing

When creating an Astra DB document store, you may see a warning similar to the following:

> Astra DB collection '...' is detected as having indexing turned on for all fields (either created manually or by older versions of this plugin). This implies stricter limitations on the amount of text each string in a document can store. Consider indexing anew on a fresh collection to be able to store longer texts.

or,

> Astra DB collection '...' is detected as having the following indexing policy: {...}. This does not match the requested indexing policy for this object: {...}. In particular, there may be stricter limitations on the amount of text each string in a document can store. Consider indexing anew on a fresh collection to be able to store longer texts.


The reason for the warning is that the requested collection already exists on the database, and it is configured to [index all of its fields for search](https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#the-indexing-option), possibly implicitly, by default. When the Haystack object tries to create it, it attempts to enforce, instead, an indexing policy tailored to the prospected usage: this is both to enable storing very long texts and to avoid indexing fields that will never be used in filtering a search (indexing those would also have a slight performance cost for writes).

Typically there are two reasons why you may encounter the warning:

1. you have created a collection by other means than letting this component do it for you: for example, through the Astra UI, or using AstraPy's `create_collection` method of class `Database` directly;
2. you have created the collection with an older version of the plugin.

Keep in mind that this is a warning and your application will continue running just fine, as long as you don't store very long texts.
However, should you need to add to the document store, for example, a document with a very long textual content, you will get an indexing error from the database.

### Remediation

You have several options:

- you can ignore the warning because you know your application will never need to store very long textual contents;
- if you can afford populating the collection anew, you can drop it and re-run the Haystack application: the collection will be created with the optimized indexing settings. **This is the recommended option, when possible**.
