# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py                  |       36 |        2 |        4 |        1 |     92% |     73-74 |
| src/haystack\_integrations/components/retrievers/elasticsearch/elasticsearch\_hybrid\_retriever.py |       75 |        0 |       14 |        3 |     97% |340-\>344, 344-\>348, 348-\>352 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py             |       35 |        2 |        4 |        1 |     92% |     72-73 |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py                   |       53 |        2 |       14 |        2 |     94% |  129, 184 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py                       |      526 |      388 |      152 |        5 |     24% |142-143, 179-\>exit, 273-275, 310, 312, 322-323, 331-333, 339-360, 366-385, 396-403, 414-421, 432-446, 459, 494-548, 572-616, 629, 638, 658-669, 683-708, 725-761, 774-789, 802-817, 831-852, 866-887, 909-944, 966-1004, 1024-1044, 1064-1087, 1097-1101, 1111-1116, 1129-1133, 1140-1148, 1166-1172, 1186-1211, 1229-1253, 1281-1287, 1313-1319, 1326, 1343-1345, 1355-1362, 1372-1379, 1404-1446, 1471-1513, 1529-1542, 1558-1571 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                               |      135 |       50 |       72 |       12 |     63% |15-16, 19, 48, 51, 66-75, 79-97, 106, 110-115, 117-118, 128, 132-137, 139-140, 150, 152-159, 161-162, 168-169, 174-177 |
| **TOTAL**                                                                                          |  **860** |  **444** |  **260** |   **24** | **46%** |           |


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