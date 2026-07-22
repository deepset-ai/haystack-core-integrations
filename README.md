# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       40 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       79 |        0 |       14 |        3 |     97% |341-\>345, 345-\>349, 349-\>353 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_hybrid\_retriever.py     |       45 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/inference\_sparse\_retriever.py     |       42 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       57 |        2 |       14 |        2 |     94% |  141, 196 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      639 |      352 |      212 |       21 |     44% |176-177, 236, 353, 355, 365-366, 374-376, 382-413, 419-448, 459-466, 477-484, 498, 501-\>529, 560, 590-591, 636-637, 640-641, 687-\>692, 689-690, 692-\>695, 707-\>706, 733-752, 780-\>785, 782-783, 785-\>788, 798-\>797, 824-842, 857, 866, 886-888, 926-927, 944-980, 993-1008, 1021-1036, 1050-1071, 1085-1106, 1128-1163, 1185-1223, 1243-1263, 1283-1306, 1560-1564, 1574-1579, 1592-1596, 1603-1611, 1629-1635, 1649-1674, 1692-1716, 1744-1750, 1776-1782, 1789, 1806-1808, 1818-1825, 1835-1842, 1867-1909, 1934-1976, 1992-2005, 2021-2034 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |        0 |       72 |        0 |    100% |           |
| **TOTAL**                                                                                          | **1115** |  **354** |  **336** |   **26** | **67%** |           |


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