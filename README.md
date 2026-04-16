# Repository Coverage (weaviate-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate-combined/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       32 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |        4 |       66 |        2 |     97% |22-23, 247-248 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      634 |       43 |      200 |       19 |     93% |299-\>exit, 308-\>exit, 489, 535, 537, 742-\>752, 797-799, 807-809, 958, 976-977, 987, 1005-1006, 1021-\>1003, 1058-\>1039, 1143-1145, 1182, 1233, 1239-\>exit, 1295-1300, 1368-1369, 1386-1387, 1414, 1434-\>1437, 1443-1447, 1450-1453, 1461-1466 |
| **TOTAL**                                                                         |  **993** |   **47** |  **290** |   **21** | **95%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-weaviate-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-weaviate-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-weaviate-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.