# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      743 |      454 |      190 |       27 |     38% |276-\>exit, 281-\>exit, 322, 325-\>exit, 330-\>exit, 371, 383-392, 403-412, 422, 426-429, 433-\>435, 441-\>exit, 459, 463, 467-\>469, 475-\>exit, 483-503, 509-547, 553-597, 606-614, 624-632, 641-684, 693-723, 730-764, 775-786, 795-811, 825-848, 863-886, 892-903, 922-958, 979-1015, 1023-1034, 1047-1058, 1069-1076, 1084-1091, 1112-1140, 1160-1188, 1199-1238, 1249-1288, 1296-1314, 1332-1349, 1361-1378, 1392-1393, 1395-1399, 1418-\>1422, 1431, 1460-1474, 1490-1505, 1508-1516, 1526-1539, 1549-1564, 1586, 1599-1618, 1655-1667, 1689-1703, 1743-1744, 1753-1756, 1776-1788, 1799-1811, 1821-1877, 1897, 1909-1929, 1941-1963, 1984-2021, 2034-2036, 2065-2088, 2117-2143 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |       44 |       72 |        8 |     74% |91-\>94, 131, 203-215, 220-227, 229-230, 235-247, 251-263, 268-269, 275-276, 286, 293 |
| **TOTAL**                                                                         | **1049** |  **507** |  **290** |   **44** | **51%** |           |


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