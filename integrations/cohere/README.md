# cohere-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/cohere-haystack.svg)](https://pypi.org/project/cohere-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cohere-haystack.svg)](https://pypi.org/project/cohere-haystack)

`cohere-haystack` integrates [Cohere](https://cohere.ai/) capabilities into [Haystack](https://github.com/deepset-ai/haystack) (2.x) pipelines. This package allows seamless use of Cohere's large language models (LLMs) for generating text, embedding text, and re-ranking search results. 

-----

**Table of Contents**

- [cohere-haystack](#cohere-haystack)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Generators](#generators)
    - [Embedders](#embedders)
    - [Rankers](#rankers)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- Easy integration with Haystack pipelines
- Use Cohere's powerful LLMs for text generation
- Embed text for similarity search
- Re-rank search results based on relevance
- Track model usage and performance

## Installation

To install `cohere-haystack`, run the following command:

```sh
pip install cohere-haystack
```

## Usage

### Generators

To use the CohereChatGenerator for text generation in your Haystack pipeline, add the `CohereChatGenerator` to your pipeline.

Here's an example using corrective loops with JSON schema validation:

```python
from typing import List

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.cohere import CohereChatGenerator

person_schema = {
    "type": "object",
    "properties": {
        "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
    },
    "required": ["first_name", "last_name", "nationality"]
}

# Initialize a pipeline
pipe = Pipeline()

# Add components to the pipeline
pipe.add_component("joiner", BranchJoiner(List[ChatMessage]))
pipe.add_component("fc_llm", CohereChatGenerator(model="command-r"))
pipe.add_component("validator", JsonSchemaValidator(json_schema=person_schema))
pipe.add_component("adapter", OutputAdapter("{{chat_message}}", List[ChatMessage])),
# And connect them
pipe.connect("adapter", "joiner")
pipe.connect("joiner", "fc_llm")
pipe.connect("fc_llm.replies", "validator.messages")
pipe.connect("validator.validation_error", "joiner")

result = pipe.run(data={"adapter": {"chat_message": [ChatMessage.from_user("Create json from Peter Parker")]}})

print(result["validator"]["validated"])
```
> [ChatMessage(content='{"first_name":"Peter","last_name":"Parker","nationality":"American"}', role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'command-r', 'usage': 464, 'index': 0, 'finish_reason': 'COMPLETE', 'documents': None, 'citations': None})]
> 
In this example, the pipeline uses `CohereChatGenerator` to generate JSON text based on a schema and validate the output. If the output does not match the schema, the pipeline will loop back to the generator for correction.

### Embedders

To use Cohere's embedding capabilities, add the `CohereTextEmbedder` and `CohereDocumentEmbedder` to your pipeline.

Here's an example:

```python
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.cohere.text_embedder import CohereTextEmbedder
from haystack_integrations.components.embedders.cohere.document_embedder import CohereDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities")
]

document_embedder = CohereDocumentEmbedder()
documents_with_embeddings = document_embedder.run(documents)['documents']
document_store.write_documents(documents_with_embeddings)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", CohereTextEmbedder())
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "Who lives in Berlin?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result['retriever']['documents'][0])
```

> Calculating embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.88it/s]
Document(id=62fad790ad2af927af9432c87330ed2ea5e31332cdec8e9d6235a5105ab0aaf5, content: 'My name is Wolfgang and I live in Berlin', score: 0.7897701476945549)
> 
In this example, documents are embedded using `CohereDocumentEmbedder`, stored in an in-memory document store, and retrieved using `InMemoryEmbeddingRetriever` based on query embeddings generated by `CohereTextEmbedder`.

### Rankers

To use Cohere's ranking capabilities, add the `CohereRanker` to your pipeline.

Here's an example:

```python
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.rankers.cohere import CohereRanker

# Note: Set your API key by running the below command in your terminal
# export CO_API_KEY="<your Cohere API key>"

docs = [
    Document(content="Paris is in France"),
    Document(content="Berlin is in Germany"),
    Document(content="Lyon is in France"),
]
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)
ranker = CohereRanker(model="rerank-english-v2.0")

document_ranker_pipeline = Pipeline()
document_ranker_pipeline.add_component(instance=retriever, name="retriever")
document_ranker_pipeline.add_component(instance=ranker, name="ranker")

document_ranker_pipeline.connect("retriever.documents", "ranker.documents")

query = "Cities in France"
res = document_ranker_pipeline.run(
    data={"retriever": {"query": query}, "ranker": {"query": query, "top_k": 2}}
)

print(res)
```

>{'ranker': {'documents': [Document(id=082ef4f19ffd14324bd316902c11d3b44a3cfe820bcc88281c88c92452616300, content: 'Lyon is in France', score: 0.9809856), Document(id=4583a7ddf7396ba413dd877de7b60f44e3512e2f3b1187dd4de32618e03b3d22, content: 'Paris is in France', score: 0.9806549)]}}


In this example, documents are first retrieved using `InMemoryBM25Retriever` and then re-ranked based on relevance to the query using `CohereRanker`.

## Contributing

`hatch` is the best way to interact with this project. To install it, run:
```sh
pip install hatch
```

With `hatch` installed, to run all the tests:
```sh
hatch run test
```
> Note: Integration tests will be skipped unless the environment variable `COHERE_API_KEY` is set. The API key needs to be valid in order to pass the tests.

To only run unit tests:
```sh
hatch run test -m "not integration"
```

To only run embedders tests:
```sh
hatch run test -m "embedders"
```

To only run generators tests:
```sh
hatch run test -m "generators"
```

To only run ranker tests:
```sh
hatch run test -m "ranker"
```

Markers can be combined. For example, you can run only integration tests for embedders with:
```sh
hatch run test -m "integration and embedders"
```

To run the linters `ruff` and `mypy`:
```sh
hatch run lint:all
```

## License

`cohere-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.