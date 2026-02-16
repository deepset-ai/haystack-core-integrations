# faiss-haystack

This package provides a [FAISS](https://github.com/facebookresearch/faiss) document store for [Haystack](https://github.com/deepset-ai/haystack).

## Installation

```bash
pip install faiss-haystack
```

## Usage

```python
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

document_store = FAISSDocumentStore(index_path="my_index")
```
