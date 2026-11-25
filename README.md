# Repository Coverage (qdrant-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/multivector\_retriever.py |       44 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py              |      130 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/utils.py                  |       17 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py                  |       51 |        0 |       22 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py             |      874 |       33 |      252 |       24 |     95% |471-472, 530-531, 572-573, 614, 629-631, 649, 664-666, 771-\>768, 792-\>787, 850, 870-\>856, 890-892, 915, 934-\>920, 938, 954-956, 1151-\>1141, 1195-\>1185, 1233-\>1223, 1271-\>1261, 1311-\>1301, 1355-\>1345, 1409-\>1399, 1465-\>1455, 2313-2314, 2332-2334, 2782-2783, 2842-2843, 2850-2851 |
| src/haystack\_integrations/document\_stores/qdrant/filters.py                     |      126 |        5 |       60 |        6 |     94% |48-\>40, 52-\>40, 74, 86, 105, 133-134 |
| **TOTAL**                                                                         | **1242** |   **38** |  **358** |   **30** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-qdrant-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.