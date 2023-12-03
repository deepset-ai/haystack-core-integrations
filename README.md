# Haystack 2.x Core Integrations

This repository contains integrations to extend the capabilities of [Haystack](https://github.com/deepset-ai/haystack) version 2.0 and
onwards. The code in this repo is maintained by [deepset](https://www.deepset.ai), see each integration's `README` file for details around installation, usage and support.

| Package                                                                         | Type                | PyPi Package                                                                                                                                             | Status                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [chroma-haystack](integrations/chroma/)                                         | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/chroma-haystack.svg)](https://pypi.org/project/chroma-haystack)                                         | [![Test / chroma](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml)                                                               |
| [cohere-haystack](integrations/cohere/)                                         | Generator           | [![PyPI - Version](https://img.shields.io/pypi/v/chroma-haystack.svg)](https://pypi.org/project/cohere-haystack)                                         | [![Test / cohere](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/cohere.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/cohere.yml)                                                               |
| [elasticsearch-haystack](integrations/elasticsearch/)                           | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)                           | [![Test / elasticsearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml)                                          |
| [gradient-haystack](integrations/gradient/)                                     | Embedder, Generator | [![PyPI - Version](https://img.shields.io/pypi/v/gradient-haystack.svg)](https://pypi.org/project/gradient-haystack)                                     | [![Test / gradient](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml)                                                         |
| [instructor-embedders-haystack](integrations/instructor-embedders/)             | Embedder            | [![PyPI - Version](https://img.shields.io/pypi/v/instructor-embedders-haystack.svg)](https://pypi.org/project/instructor-embedders-haystack)             | [![Test / instructor-embedders](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml)                     |
| [opensearch-haystack](integrations/opensearch/)                                 | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/opensearch-haystack.svg)](https://pypi.org/project/opensearch-haystack)                                 | [![Test / opensearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml)                                                   |
| [unstructured-fileconverter-haystack](integrations/unstructured/fileconverter/) | File converter      | [![PyPI - Version](https://img.shields.io/pypi/v/unstructured-fileconverter-haystack.svg)](https://pypi.org/project/unstructured-fileconverter-haystack) | [![Test / unstructured / fileconverter](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured_fileconverter.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured_fileconverter.yml) |

## Contributing

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.
