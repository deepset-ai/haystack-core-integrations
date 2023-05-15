# Haystack Extras

This repository contains extra components for [Haystack][haystack-repo], see the component `README.md` for detailed installation and usage instructions.

This is the list of packages currently part of the repo.

| Package                                               | Type           | PyPi Package                                                                                                                         | Status                                                                            |
| ----------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| [text2speech](./nodes/text2speech/)                   | Node           | [![PyPI - Version](https://img.shields.io/pypi/v/farm-haystack-text2speech.svg)](https://pypi.org/project/farm-haystack-text2speech) | [![Test / text2speech][text2speechbadge]][text2speech]                            |
| [mongodb-documentstore](stores/mongodb-documentstore) | Document Store | n/a                                                                                                                                  | [![Test / MongoDBDocumentStore][mongodbdocumentstorebadge]][mongodbdocumentstore] |
| [milvus-documentstore](stores/milvus-documentstore)   | Document Store | n/a                                                                                                                                  |                                                                                   |



## Contributing

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.

[haystack-repo]: https://github.com/deepset-ai/haystack
[text2speechbadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml/badge.svg
[text2speech]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml
[mongodbdocumentstorebadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_mongodb_document_store.yml/badge.svg
[mongodbdocumentstore]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_mongodb_document_store.yml
