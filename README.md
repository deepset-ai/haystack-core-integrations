# Haystack 2.x Core Integrations

This repository contains integrations to extend the capabilities of [Haystack](https://github.com/deepset-ai/haystack) version 2.0 and
onwards. The code in this repo is maintained by [deepset](https://www.deepset.ai), see each integration's `README` file for details around installation, usage and support.

## Contributing

You will need `hatch` to work on or create new integrations. Run `pip install hatch` to install it.

### Local development

All the integrations are self contained, so the first step before working on one is to `cd` into the proper folder.
For example, to work on the Chroma Document Store, from the root of the repo:

```sh
$ cd integrations/chroma
```

From there, you can run the tests with `hatch`, that will take care of setting up an isolated Python environment:

```sh
hatch run test
```

Similarly, to run the linters:

```sh
hatch run lint:all
```

### Create a new integration

> Core integrations follow the naming convention `PREFIX-haystack`, where `PREFIX` can be the name of the technology
> you're integrating Haystack with. For example, a deepset integration would be named as `deepset-haystack`.

To create a new integration, from the root of the repo change directory into `integrations`:

```sh
cd integrations
```

From there, use `hatch` to create the scaffold of the new integration:

```sh
$ hatch --config hatch.toml new -i
Project name: deepset-haystack
Description []: An example integration, this text can be edited later

deepset-haystack
├── src
│   └── deepset_haystack
│       ├── __about__.py
│       └── __init__.py
├── tests
│   └── __init__.py
├── LICENSE.txt
├── README.md
└── pyproject.toml
```

## Inventory

| Package                                                             | Type                | PyPi Package                                                                                                                                             | Status                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [astra-haystack](integrations/astra/)                               | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/astra-haystack.svg)](https://pypi.org/project/astra-haystack)                                           | [![Test / astra](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/astra.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/astra.yml)                                              |
| [amazon-bedrock-haystack](integrations/amazon-bedrock/)             | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/amazon-bedrock-haystack.svg)](https://pypi.org/project/amazon-bedrock-haystack)                         | [![Test / amazon_bedrock](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/amazon_bedrock.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/amazon_bedrock.yml)                   |
| [chroma-haystack](integrations/chroma/)                             | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/chroma-haystack.svg)](https://pypi.org/project/chroma-haystack)                                         | [![Test / chroma](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml)                                           |
| [cohere-haystack](integrations/cohere/)                             | Embedder, Generator | [![PyPI - Version](https://img.shields.io/pypi/v/cohere-haystack.svg)](https://pypi.org/project/cohere-haystack)                                         | [![Test / cohere](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/cohere.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/cohere.yml)                                           |
| [elasticsearch-haystack](integrations/elasticsearch/)               | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)                           | [![Test / elasticsearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml)                      |
| [google-ai-haystack](integrations/google_ai/)                       | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/google-ai-haystack.svg)](https://pypi.org/project/google-ai-haystack)                                   | [![Test / google-ai](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/google_ai.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/google_ai.yml)                                  |
| [google-vertex-haystack](integrations/google_vertex/)               | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/google-vertex-haystack.svg)](https://pypi.org/project/google-vertex-haystack)                           | [![Test / google-vertex](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/google_vertex.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/google_vertex.yml)                      |
| [gradient-haystack](integrations/gradient/)                         | Embedder, Generator | [![PyPI - Version](https://img.shields.io/pypi/v/gradient-haystack.svg)](https://pypi.org/project/gradient-haystack)                                     | [![Test / gradient](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml)                                     |
| [instructor-embedders-haystack](integrations/instructor_embedders/) | Embedder            | [![PyPI - Version](https://img.shields.io/pypi/v/instructor-embedders-haystack.svg)](https://pypi.org/project/instructor-embedders-haystack)             | [![Test / instructor-embedders](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml) |
| [jina-haystack](integrations/jina/)                                 | Embedder            | [![PyPI - Version](https://img.shields.io/pypi/v/jina-haystack.svg)](https://pypi.org/project/jina-haystack)                                             | [![Test / jina](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/jina.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/jina.yml)                                                 |
| [llama-cpp-haystack](integrations/llama_cpp/)                       | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/ollama-haystack.svg?color=orange)](https://pypi.org/project/llama-cpp-haystack)                         | [![Test / llama-cpp](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/llama_cpp.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/llama_cpp.yml)                                  |
| [ollama-haystack](integrations/ollama/)                             | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/ollama-haystack.svg?color=orange)](https://pypi.org/project/ollama-haystack)                            | [![Test / ollama](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/ollama.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/ollama.yml)                                           |
| [opensearch-haystack](integrations/opensearch/)                     | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/opensearch-haystack.svg)](https://pypi.org/project/opensearch-haystack)                                 | [![Test / opensearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml)                               |
| [pinecone-haystack](integrations/pinecone/)                         | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/pinecone-haystack.svg?color=orange)](https://pypi.org/project/pinecone-haystack)                        | [![Test / pinecone](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/pinecone.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/pinecone.yml)                                     |
| [pgvector-haystack](integrations/pgvector/)                         | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/pgvector-haystack.svg?color=orange)](https://pypi.org/project/pgvector-haystack)                        | [![Test / pgvector](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/pgvector.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/pgvector.yml)                                     |
| [qdrant-haystack](integrations/qdrant/)                             | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/qdrant-haystack.svg?color=orange)](https://pypi.org/project/qdrant-haystack)                            | [![Test / qdrant](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/qdrant.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/qdrant.yml)                                           |
| [unstructured-fileconverter-haystack](integrations/unstructured/)   | File converter      | [![PyPI - Version](https://img.shields.io/pypi/v/unstructured-fileconverter-haystack.svg)](https://pypi.org/project/unstructured-fileconverter-haystack) | [![Test / unstructured](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured.yml)                         |
| [uptrain-haystack](integrations/uptrain/)                           | Evaluator           | [![PyPI - Version](https://img.shields.io/pypi/v/uptrain-haystack.svg)](https://pypi.org/project/uptrain-haystack)                                       | [![Test / uptrain](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/uptrain.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/uptrain.yml)                                        |
| [amazon-sagemaker-haystack](integrations/amazon_sagemaker/)         | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/amazon-sagemaker-haystack.svg)](https://pypi.org/project/amazon-sagemaker-haystack)                     | [![Test / amazon_sagemaker](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/amazon_sagemaker.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/amazon_sagemaker.yml)             |
