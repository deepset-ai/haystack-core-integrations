# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       40 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       79 |        0 |       14 |        3 |     97% |341-\>345, 345-\>349, 349-\>353 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_dense\_retriever.py      |       43 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_hybrid\_retriever.py     |       45 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       42 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       57 |        2 |       14 |        2 |     94% |  141, 196 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      692 |      385 |      230 |       21 |     43% |176-177, 236, 353, 355, 365-366, 374-376, 382-413, 419-448, 464-471, 487-494, 508, 511-\>539, 570, 600-601, 646-647, 650-651, 697-\>702, 699-700, 702-\>705, 717-\>716, 743-762, 790-\>795, 792-793, 795-\>798, 808-\>807, 834-852, 867, 876, 896-898, 936-937, 954-990, 1005-1020, 1035-1050, 1066-1087, 1103-1124, 1146-1181, 1203-1241, 1261-1281, 1301-1324, 1641-1648, 1669-1677, 1687-1691, 1701-1706, 1719-1723, 1730-1738, 1756-1762, 1776-1801, 1819-1843, 1871-1877, 1903-1909, 1916, 1933-1935, 1945-1952, 1962-1969, 1980-2003, 2014-2027, 2031-2035, 2046-2059, 2067-2080, 2118-2127, 2155-2164, 2180-2193, 2209-2222 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      147 |        0 |       78 |        0 |    100% |           |
| **TOTAL**                                                                                          | **1223** |  **387** |  **366** |   **26** | **67%** |           |


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