# fastembed-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#Usage)
- [License](#license)

## Installation

```console
pip install fastembed-haystack
```

## Usage

You can use `FastembedTextEmbedder` and `FastembedDocumentEmbedder` by importing as:

```python
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

text = "fastembed is supported by and maintained by Qdrant."
text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5"
)
text_embedder.warm_up()
embedding = text_embedder.run(text)["embedding"]
```

```python
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
from haystack import Document

embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5",
)
embedder.warm_up()
doc = Document(content="fastembed is supported by and maintained by Qdrant.", meta={"long_answer": "no",})
result = embedder.run(documents=[doc])
```

You can use `FastembedSparseTextEmbedder` and `FastembedSparseDocumentEmbedder` by importing as:

```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

text = "fastembed is supported by and maintained by Qdrant."
text_embedder = FastembedSparseTextEmbedder(
    model="prithivida/Splade_PP_en_v1"
)
text_embedder.warm_up()
embedding = text_embedder.run(text)["sparse_embedding"]
```

```python
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack import Document

embedder = FastembedSparseDocumentEmbedder(
    model="prithivida/Splade_PP_en_v1",
)
embedder.warm_up()
doc = Document(content="fastembed is supported by and maintained by Qdrant.", meta={"long_answer": "no",})
result = embedder.run(documents=[doc])
```

You can use `FastembedRanker` by importing as:

```python
from haystack import Document

from haystack_integrations.components.rankers.fastembed import FastembedRanker

query = "Who is maintaining Qdrant?"
documents = [
    Document(
        content="This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc."
    ),
    Document(content="fastembed is supported by and maintained by Qdrant."),
]

ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
ranker.warm_up()
reranked_documents = ranker.run(query=query, documents=documents)["documents"]

print(reranked_documents[0])

# Document(id=...,
#  content: 'fastembed is supported by and maintained by Qdrant.',
#  score: 5.472434997558594..)
```

## License

`fastembed-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
