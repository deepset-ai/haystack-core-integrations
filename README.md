# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      739 |      484 |      190 |       17 |     34% |276-\>exit, 281-\>exit, 322, 325-\>exit, 330-\>exit, 371, 380-410, 421-453, 460-480, 486-524, 530-574, 583-591, 601-609, 618-661, 670-700, 707-741, 752-763, 772-788, 802-825, 840-863, 869-880, 899-935, 956-992, 1000-1013, 1023-1036, 1046-1053, 1061-1068, 1082-1114, 1124-1156, 1167-1206, 1217-1256, 1264-1282, 1300-1317, 1329-1346, 1360-1361, 1363-1367, 1386-\>1390, 1399, 1428-1442, 1458-1473, 1476-1484, 1494-1507, 1517-1532, 1554, 1567-1586, 1623-1635, 1657-1671, 1711-1712, 1721-1724, 1744-1756, 1767-1779, 1789-1845, 1865, 1877-1897, 1909-1931, 1952-1989, 2002-2004, 2033-2056, 2085-2111 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |       44 |       72 |        8 |     74% |91-\>94, 131, 203-215, 220-227, 229-230, 235-247, 251-263, 268-269, 275-276, 286, 293 |
| **TOTAL**                                                                         | **1045** |  **537** |  **290** |   **34** | **48%** |           |


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