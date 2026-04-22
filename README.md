# Repository Coverage (elasticsearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       36 |        2 |        4 |        1 |     92% |     74-75 |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       75 |        0 |       14 |        3 |     97% |340-\>344, 344-\>348, 348-\>352 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       35 |        2 |        4 |        1 |     92% |     73-74 |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       38 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       53 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      572 |       46 |      170 |       29 |     90% |144-145, 312, 314, 384-\>376, 399-400, 417-418, 437, 486, 562-563, 566-567, 613-\>618, 693-\>698, 734, 741-742, 884-886, 912-914, 940-942, 975-977, 1010-1012, 1035-1036, 1057, 1062, 1094-1095, 1150-1151, 1153-\>1156, 1354-\>1353, 1391-\>1390, 1393-\>1390, 1427, 1463-1464, 1469, 1702, 1710, 1733-\>1736 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |        5 |       72 |        4 |     96% |16-17, 52, 71, 74 |
| **TOTAL**                                                                                          |  **979** |   **55** |  **288** |   **38** | **93%** |           |


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