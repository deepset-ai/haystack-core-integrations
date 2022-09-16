# haystack_mongodb_documentstore

[![PyPI - Version](https://img.shields.io/pypi/v/haystack-mongodb-documentstore.svg)](https://pypi.org/project/mongodb-documentstore)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mongodb-documentstore.svg)](https://pypi.org/project/mongodb-documentstore)

-----

**Table of Contents**

- [haystack_mongodb_documentstore](#haystack_mongodb_documentstore)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

```console
pip install haystack-mongodb-documentstore
```

## Usage

```python
from mongodb_documentstore import MongoDBDocumentStore


pipe = ExtractiveQAPipeline(
    FARMReader(model_name_or_path="deepset/roberta-base-squad2"),
    TfidfRetriever(document_store=MongoDBDocumentStore())
)

prediction = pipe.run(
    query="Who is the father of Arya Stark?",
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
)
print_answers(prediction, details="minimum")
```

## License

`haystack-mongodb-documentstore` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
