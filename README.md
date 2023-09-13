# Haystack Extras

This repository contains extra components for [Haystack][haystack-repo], maintained by [deepset](https://www.deepset.ai) on a best-effort basis. 
See each component's `README` file for detailed installation and usage instructions.

This is the list of packages currently hosted in this repo.

| Package                                             | Type           | PyPi Package                                                                                      | Status                                                 |
| --------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [text2speech](nodes/text2speech/)                   | Node           | [![PyPI - Version](https://img.shields.io/pypi/v/farm-haystack-text2speech.svg)][text2speechPypi] | [![Test / text2speech][text2speechbadge]][text2speech] |
| [milvus-documentstore](stores/milvus-documentstore) | Document Store | n/a                                                                                               | [![Tests / MilvusDocumentStore][milvus_badge]][milvus] |

## Contributing

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.



[haystack-repo]: https://github.com/deepset-ai/haystack
[text2speechbadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml/badge.svg
[text2speech]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml
[text2speechPypi]: https://pypi.org/project/farm-haystack-text2speech
[milvus_badge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_milvus_document_store.yml/badge.svg
[milvus]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_milvus_document_store.yml
