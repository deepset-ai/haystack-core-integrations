# solr-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/solr-haystack.svg)](https://pypi.org/project/solr-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/solr-haystack.svg)](https://pypi.org/project/solr-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/solr)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/solr/CHANGELOG.md)

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [How documents are stored](#how-documents-are-stored)
- [Development](#development)
- [Contributing](#contributing)

## Installation

```console
pip install solr-haystack
```

## Usage

```python
from haystack import Document
from haystack_integrations.document_stores.solr import SolrDocumentStore
from haystack_integrations.components.retrievers.solr import SolrBM25Retriever

document_store = SolrDocumentStore(url="http://localhost:8983/solr", core="haystack")
document_store.write_documents([Document(content="Apache Solr is a search platform.")])

retriever = SolrBM25Retriever(document_store=document_store)
print(retriever.run(query="search platform")["documents"])
```

Three retrievers are available:

| Component | Search |
|---|---|
| `SolrBM25Retriever` | keyword search using Solr's BM25 similarity |
| `SolrEmbeddingRetriever` | dense vector search using `DenseVectorField` and `{!knn}` |
| `SolrHybridRetriever` | both of the above, fused with reciprocal rank fusion |

Every method has an `async` counterpart (`write_documents_async`, `filter_documents_async`,
`run_async`, and so on).

## Requirements

**Solr 9.6 or newer.** Dependable k-NN pre-filtering arrived in 9.6
([SOLR-16858](https://issues.apache.org/jira/browse/SOLR-16858)); on earlier versions an embedding
search combined with filters silently returns fewer documents than requested. The document store
checks the server version on first use and refuses to run below that.

By default the document store expects the core to exist already and manages its schema:

- `create_core=False` (default) - create the core yourself, for example with `bin/solr create -c
  haystack`. Setting `create_core=True` makes the store create it through the CoreAdmin API, which
  only works if the configset named by `config_set` exists under `<solr_home>/configsets`; a stock
  installation keeps `_default` elsewhere, so this is off by default.
- `manage_schema=True` (default) - on first use the store adds the fields it needs and turns off
  Solr's schemaless field guessing. Set it to `False` to manage the schema yourself; the fields
  required in that case are listed below.

Authentication uses HTTP basic auth, read from `SOLR_USERNAME` and `SOLR_PASSWORD` by default. Pass
`auth=None` for an unauthenticated server, and `SOLR_URL` sets the default base URL.

### Cores that share a configset share a schema

A core created from a configset (`bin/solr create -c mycore -d _default`, or `create_core=True`) does
not get its own `conf` directory: it references the configset, and the managed schema and config
overlay live there. Several such cores therefore share one schema.

That matters for `embedding_dim`, because a Solr vector field's dimension is fixed when the field is
created. Two cores sharing a configset cannot use different embedding dimensions - the second store to
start up will refuse to run and say so. Give each dimension its own configset (`config_set=...`), or
create the cores with their own `conf` directories, if you need to mix dimensions on one server.

## How documents are stored

Solr fixes a field's type when the field is created, while Haystack metadata is an arbitrary
dictionary whose value types are only known at write time. Each metadata entry is therefore stored in
a field whose name encodes the Python type of the value:

| Haystack | Solr field | Solr type |
|---|---|---|
| `id` | `id` | `string` (uniqueKey) |
| `content` | `content` | `text_general` |
| `embedding` | `embedding` | `DenseVectorField` |
| `blob` | `blob` | `string` (JSON) |
| `meta["page"] = "100"` | `meta_s_page` | `string` |
| `meta["page"] = 100` | `meta_l_page` | `plong` |
| `meta["rating"] = 0.5` | `meta_d_rating` | `pdouble` |
| `meta["flag"] = True` | `meta_b_flag` | `boolean` |
| `meta["tags"] = ["a"]` | `meta_ss_tags` | `string`, multiValued |
| anything else | `meta_j_<key>` | `string` (JSON) |

Two consequences worth knowing:

- Metadata round-trips with its type intact, so `{"page": "100"}` never comes back as `{"page": 100}`,
  and the int `1`, the str `"1"`, the float `1.0` and `True` stay four distinct values.
- Metadata keys become Solr field names, so they must consist of letters, digits and underscores.
  Writing a document with any other key raises a `ValueError` naming the offending keys rather than
  silently rewriting them.

Solr has no sparse vector field, so `Document.sparse_embedding` is ignored with a warning.

`update_by_filter` rewrites whole documents rather than issuing atomic updates, because the type code
is part of the field name: setting one field would leave the previous value behind in another whenever
a metadata value changes type.

## Development

Integration tests need a running Solr. From this directory:

```console
docker compose up -d --wait
```

The container copies the `_default` configset into the Solr home so that tests can create a throwaway
core per test, which is what makes running them in parallel safe. Then:

```console
hatch run fmt          # format and lint
hatch run test:types   # mypy
hatch run test:unit    # unit tests, no Solr needed
hatch run test:integration
hatch run test:all -n 4
```

To check the Solr 9 floor, override the image:

```console
SOLR_IMAGE=solr:9.10.1 docker compose up -d --wait
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
