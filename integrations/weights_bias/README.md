# weights_biases-haystack

[![PyPI - License](https://img.shields.io/pypi/l/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)
[![PyPI - Version](https://img.shields.io/pypi/v/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)

---

**Table of Contents**

- [weights_bias-haystack](#weights_bias-haystack)
  - [Installation](#installation)
  - [Example](#example)
  - [License](#license)


## Installation

```console
pip install weights_biases-haystack
```

## Weave by Weights & Biases

In this example, we will demonstrate how to use Weave by Weights & Biases to trace the results of a hybrid pipeline 
in Haystack. 

### Example 

You need to have a Weave account to use this feature. You can sign up for free at https://wandb.ai/site.

You then need to set the `WANDB_API_KEY:` environment variable with your Weights & Biases API key. You should find 
your API key in https://wandb.ai/home when you are logged in. 

Now let's create a pipeline that uses a hybrid retrieval strategy, combining BM25 and embeddings, and trace the results
in Weave.

Create a document store with some example documents and index them. 

```python
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

document_store = InMemoryDocumentStore()
documents = [
    Document(content="The Eiffel Tower is located in Paris, France."),
    Document(content="The capital of Germany is Berlin."),
    Document(content="The Colosseum is located in Rome, Italy."),
    Document(content="Paris is the capital of France."),
    Document(content="The Louvre is located in Paris, France."),
    Document(content="The Brandenburg Gate is located in Berlin, Germany."),
]

embedder = SentenceTransformersDocumentEmbedder()
embedder.warm_up()
embed_docs = embedder.run(documents)
document_store.write_documents(embed_docs['documents'])
```

Now let's create a pipeline and connect it to Weave, to do so we add a `WeaveConnector` component to the pipeline, 
which will send the results to Weave.

```python
from haystack import Document
from haystack import Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.connectors import WeaveConnector

text_embedder = SentenceTransformersTextEmbedder()
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)
document_joiner = DocumentJoiner()
ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

hybrid_retrieval = Pipeline()
hybrid_retrieval.add_component("text_embedder", text_embedder)
hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
hybrid_retrieval.add_component("document_joiner", document_joiner)
hybrid_retrieval.add_component("ranker", ranker)
hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "document_joiner")
hybrid_retrieval.connect("embedding_retriever", "document_joiner")
hybrid_retrieval.connect("document_joiner", "ranker")

connector = WeaveConnector(pipeline_name="test_pipeline")
hybrid_retrieval.add_component("connector", connector)
```

Now we can run the pipeline and trace the results in Weave. 

```python
query = "What is the capital of France?"
results = hybrid_retrieval.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}})
```

You should then head to `https://wandb.ai/<user_name>/projects` and see the complete trace for your pipeline under
the pipeline name you specified, when creating the `WeaveConnector`.


## Example with Docker

https://hub.docker.com/r/wandb/weave-trace


## License