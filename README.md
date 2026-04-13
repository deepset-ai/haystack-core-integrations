# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       36 |        2 |        4 |        1 |     92% |     73-74 |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       75 |        0 |       14 |        3 |     97% |340-\>344, 344-\>348, 348-\>352 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       35 |        2 |        4 |        1 |     92% |     72-73 |
| src/haystack\_integrations/components/retrievers/elasticsearch/sparse\_embedding\_retriever.py     |       35 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       53 |        2 |       14 |        2 |     94% |  129, 184 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      551 |      391 |      160 |        6 |     27% |143-144, 274-276, 311, 313, 323-324, 332-334, 340-361, 367-386, 397-404, 415-422, 433-447, 476, 478, 508-509, 557-611, 635-679, 692, 701, 721-732, 746-771, 788-824, 837-852, 865-880, 894-915, 929-950, 972-1007, 1029-1067, 1087-1107, 1127-1150, 1207-1211, 1221-1226, 1239-1243, 1250-1258, 1276-1282, 1296-1321, 1339-1363, 1391-1397, 1423-1429, 1436, 1453-1455, 1465-1472, 1482-1489, 1514-1556, 1581-1623, 1639-1652, 1668-1681 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |       49 |       72 |       11 |     64% |15-16, 48, 51, 66-75, 79-97, 106, 110-115, 117-118, 128, 132-137, 139-140, 150, 152-159, 161-162, 168-169, 174-177 |
| **TOTAL**                                                                                          |  **920** |  **446** |  **272** |   **24** | **49%** |           |


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