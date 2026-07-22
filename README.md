# Repository Coverage (elasticsearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       40 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       79 |        0 |       14 |        3 |     97% |341-\>345, 345-\>349, 349-\>353 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_hybrid\_retriever.py     |       45 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       42 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       57 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      639 |       52 |      212 |       31 |     90% |176-177, 353, 355, 392-394, 429-431, 445-\>437, 460-461, 478-479, 498, 560, 636-637, 640-641, 687-\>692, 780-\>785, 834, 841-842, 978-980, 1006-1008, 1034-1036, 1069-1071, 1104-1106, 1129-1130, 1151, 1156, 1188-1189, 1244-1245, 1247-\>1250, 1594-\>1593, 1631-\>1630, 1633-\>1630, 1667, 1703-1704, 1709, 1942, 1950, 1973-\>1976 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |        0 |       72 |        0 |    100% |           |
| **TOTAL**                                                                                          | **1115** |   **52** |  **336** |   **34** | **94%** |           |


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