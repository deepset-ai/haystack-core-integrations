# haystack-extras

> NOTICE: this is my Hacky Friday project, do not rely on anything in this repo at this moment!

You will need `hatch` to create new projects in this folder. Run `pip install -r requirements.txt` to install it.

| Package                                               | Type           | Status                                                                            |
| ----------------------------------------------------- | -------------- | --------------------------------------------------------------------------------- |
| [text2speech](./nodes/text2speech/)                   | Node           | [![Test / text2speech][text2speechBadge]][text2speech]                            |
| [mongodb-documentstore](stores/mongodb-documentstore) | Document Store | [![Test / MongoDBDocumentStore][MongoDBDocumentStoreBadge]][MongoDBDocumentStore] |


[text2speechBadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml/badge.svg
[text2speech]: https://github.com/deepset-ai/haystack-extras/actions/workflows/nodes_text2speech.yml
[MongoDBDocumentStoreBadge]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_mongodb_document_store.yml/badge.svg
[MongoDBDocumentStore]: https://github.com/deepset-ai/haystack-extras/actions/workflows/stores_mongodb_document_store.yml
