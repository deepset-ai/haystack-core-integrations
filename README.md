# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       36 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       75 |        0 |       14 |        3 |     97% |340-\>344, 344-\>348, 348-\>352 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_hybrid\_retriever.py     |       41 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       38 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       53 |        2 |       14 |        2 |     94% |  129, 184 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      627 |      364 |      204 |       19 |     41% |174-175, 343, 345, 355-356, 364-366, 372-403, 409-438, 449-456, 467-474, 488, 491-\>519, 550, 580-581, 626-627, 630-631, 677-\>682, 679-680, 682-\>685, 697-\>696, 723-742, 770-\>775, 772-773, 775-\>778, 788-\>787, 814-832, 847, 856, 876-878, 898-923, 940-976, 989-1004, 1017-1032, 1046-1067, 1081-1102, 1124-1159, 1181-1219, 1239-1259, 1279-1302, 1556-1560, 1570-1575, 1588-1592, 1599-1607, 1625-1631, 1645-1670, 1688-1712, 1740-1746, 1772-1778, 1785, 1802-1804, 1814-1821, 1831-1838, 1863-1905, 1930-1972, 1988-2001, 2017-2030 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |        0 |       72 |        0 |    100% |           |
| **TOTAL**                                                                                          | **1075** |  **366** |  **328** |   **24** | **65%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-elasticsearch%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.