# milvus-documentstore


## Installation

```console
pip install -e "git+https://github.com/deepset-ai/haystack-extras.git#egg=milvus_documentstore&subdirectory=stores/milvus-documentstore"
```
## Usage
The package will be available in your code under the name `milvus_documentstore`, so for example you can add a document like this:

```py
from haystack import Document
from milvus_documentstore import MilvusDocumentStore

ds = MilvusDocumentStore()
ds.write_documents([Document("Some Content")])
ds.get_all_documents()  # prints [<Document: {'content': 'foo', 'content_type': 'text', ...>]
```

## License

`milvus-documentstore` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
