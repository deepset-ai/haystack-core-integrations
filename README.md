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

| Package                                                                         | Type                | PyPi Package                                                                                                                                             | Status                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [chroma-haystack](integrations/chroma/)                                         | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/chroma-haystack.svg)](https://pypi.org/project/chroma-haystack)                                         | [![Test / chroma](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/chroma.yml)                                                               |
| [elasticsearch-haystack](integrations/elasticsearch/)                           | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)                           | [![Test / elasticsearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/elasticsearch.yml)                                          |
| [gradient-haystack](integrations/gradient/)                                     | Embedder, Generator | [![PyPI - Version](https://img.shields.io/pypi/v/gradient-haystack.svg)](https://pypi.org/project/gradient-haystack)                                     | [![Test / gradient](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/gradient.yml)                                                         |
| [instructor-embedders-haystack](integrations/instructor-embedders/)             | Embedder            | [![PyPI - Version](https://img.shields.io/pypi/v/instructor-embedders-haystack.svg)](https://pypi.org/project/instructor-embedders-haystack)             | [![Test / instructor-embedders](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/instructor_embedders.yml)                     |
| [opensearch-haystack](integrations/opensearch/)                                 | Document Store      | [![PyPI - Version](https://img.shields.io/pypi/v/opensearch-haystack.svg)](https://pypi.org/project/opensearch-haystack)                                 | [![Test / opensearch](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml)                                                   |
| [unstructured-fileconverter-haystack](integrations/unstructured/fileconverter/) | File converter      | [![PyPI - Version](https://img.shields.io/pypi/v/unstructured-fileconverter-haystack.svg)](https://pypi.org/project/unstructured-fileconverter-haystack) | [![Test / unstructured / fileconverter](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured_fileconverter.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/unstructured_fileconverter.yml) |
