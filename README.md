# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      742 |      478 |      190 |       19 |     35% |277-\>exit, 282-\>exit, 323, 326-\>exit, 331-\>exit, 372, 381-411, 422-454, 461-481, 487-525, 531-575, 584-592, 602-610, 619-662, 671-701, 708-742, 753-764, 773-789, 803-826, 841-864, 870-881, 900-936, 957-993, 1001-1014, 1024-1037, 1047-1054, 1062-1069, 1090-1118, 1138-1166, 1177-1216, 1227-1266, 1274-1292, 1310-1327, 1339-1356, 1370-1371, 1373-1377, 1396-\>1400, 1409, 1438-1452, 1468-1483, 1486-1494, 1504-1517, 1527-1542, 1564, 1577-1596, 1633-1645, 1667-1681, 1721-1722, 1731-1734, 1754-1766, 1777-1789, 1799-1855, 1875, 1887-1907, 1919-1941, 1962-1999, 2012-2014, 2043-2066, 2095-2121 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |       44 |       72 |        8 |     74% |91-\>94, 131, 203-215, 220-227, 229-230, 235-247, 251-263, 268-269, 275-276, 286, 293 |
| **TOTAL**                                                                         | **1048** |  **531** |  **290** |   **36** | **48%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-pgvector%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.