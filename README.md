# Repository Coverage (elasticsearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       36 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       75 |        0 |       14 |        3 |     97% |340-\>344, 344-\>348, 348-\>352 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_hybrid\_retriever.py     |       41 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       38 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       53 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      627 |       52 |      204 |       31 |     90% |174-175, 343, 345, 382-384, 419-421, 435-\>427, 450-451, 468-469, 488, 550, 626-627, 630-631, 677-\>682, 770-\>775, 824, 831-832, 974-976, 1002-1004, 1030-1032, 1065-1067, 1100-1102, 1125-1126, 1147, 1152, 1184-1185, 1240-1241, 1243-\>1246, 1590-\>1589, 1627-\>1626, 1629-\>1626, 1663, 1699-1700, 1705, 1938, 1946, 1969-\>1972 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |        0 |       72 |        0 |    100% |           |
| **TOTAL**                                                                                          | **1075** |   **52** |  **328** |   **34** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-elasticsearch-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.