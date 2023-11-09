# Haystack 2.x additional resources

This repository contains integrations to extend the capabilities of [Haystack](https://github.com/deepset-ai/haystack) version 2.0 and
onwards. The code in this repo is maintained by [deepset](https://www.deepset.ai), some of it on a best-effort
basis: see each folder's `README` file for details around installation, usage and support.

This is the list of packages currently hosted in this repo.

| Package                                                           | Type           | PyPi Package                                                                                                                                 | Status                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [chroma-haystack](document_stores/chroma/)                        | Document Store | [![PyPI - Version](https://img.shields.io/pypi/v/chroma-haystack.svg)](https://pypi.org/project/chroma-haystack)                             | [![Test / Document Stores / chroma](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/document_stores_chroma.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/document_stores_chroma.yml)               |
| [instructor-embedders-haystack](components/instructor-embedders/) | Embedder       | [![PyPI - Version](https://img.shields.io/pypi/v/instructor-embedders-haystack.svg)](https://pypi.org/project/instructor-embedders-haystack) | [![Test / instructor-embedders](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/components_instructor_embedders.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/components_instructor_embedders.yml) |


## Contributing

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.
