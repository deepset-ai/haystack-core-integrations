# Repository Coverage (elasticsearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch/htmlcov/index.html)

| Name                                                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py      |       36 |        2 |        4 |        1 |     92% |     73-74 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py |       35 |        2 |        4 |        1 |     92% |     72-73 |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py       |       53 |        2 |       14 |        2 |     94% |  129, 184 |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py           |      509 |      389 |      146 |        4 |     21% |134-135, 164-\>exit, 258-260, 294, 296, 306-307, 315-317, 323-344, 350-369, 380-387, 398-405, 417-423, 447-510, 534-587, 600, 609, 629-640, 654-679, 696-732, 745-760, 773-788, 802-823, 837-858, 880-915, 937-975, 995-1015, 1035-1058, 1068-1072, 1082-1087, 1100-1104, 1111-1119, 1137-1143, 1157-1182, 1200-1224, 1252-1258, 1284-1290, 1297, 1314-1316, 1326-1333, 1343-1350, 1375-1417, 1442-1484, 1500-1513, 1529-1542 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                   |      135 |       50 |       72 |       12 |     63% |15-16, 19, 48, 51, 66-75, 79-97, 106, 110-115, 117-118, 128, 132-137, 139-140, 150, 152-159, 161-162, 168-169, 174-177 |
| **TOTAL**                                                                              |  **768** |  **445** |  **240** |   **20** | **40%** |           |


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