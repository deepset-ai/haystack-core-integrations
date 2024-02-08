# fastembed-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install fastembed-haystack
```

## Usage

You can use `FastembedTextEmbedder` and `JinaDocumentEmbedder` by importing as:

```python
from fastembed_haystack.fastembed_text_embedder import FastembedTextEmbedder

text = "fastembed is supported by and maintained by Qdrant."
text_embedder = FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5"
)
embedding = text_embedder.run(text)
```

```python
from fastembed_haystack.fastembed__document_embedder import FastembedDocumentEmbedder
from haystack.dataclasses import Document

embedder = FastembedDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5",
)
embedder.warm_up()
doc = Document(content="fastembed is supported by and maintained by Qdrant.", meta={"long_answer": "no",})
result = embedder.run(documents=[doc])
```

## License

`fastembed-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
