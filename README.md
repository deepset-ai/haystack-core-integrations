# Haystack 2.x additional resources

This repository contains integrations to extend the capabilities of [Haystack][haystack-repo] version 2.0 and
onwards. The code in this repo is maintained by [deepset](https://www.deepset.ai), some of it on a best-effort
basis: see each folder's `README` file for details around installation, usage and support.

This is the list of packages currently hosted in this repo.

| Package                                                  | Type | PyPi Package                                                                                                                                 | Status                                                                                                                                                                                                                                                                     |
| -------------------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [instructor-embedders](components/instructor-embedders/) | Node | [![PyPI - Version](https://img.shields.io/pypi/v/instructor-embedders-haystack.svg)](https://pypi.org/project/instructor-embedders-haystack) | [![Test / instructor-embedders](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/components_instructor_embedders.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/components_instructor_embedders.yml) |


## Contributing

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.



[haystack-repo]: https://github.com/deepset-ai/haystack
[text2speechbadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml/badge.svg
[text2speech]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml
[text2speechPypi]: https://pypi.org/project/farm-haystack-text2speech
[milvus_badge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_milvus_document_store.yml/badge.svg
[milvus]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_milvus_document_store.yml
