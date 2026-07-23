# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      731 |      476 |      186 |       17 |     34% |276-\>exit, 281-\>exit, 322, 325-\>exit, 330-\>exit, 371, 380-410, 421-453, 460-480, 486-524, 530-574, 583-591, 601-609, 618-661, 670-700, 707-741, 752-763, 772-788, 802-825, 840-863, 869-880, 899-935, 956-987, 995-1008, 1018-1031, 1041-1048, 1056-1063, 1077-1109, 1119-1151, 1162-1201, 1212-1251, 1259-1277, 1295-1312, 1324-1341, 1355-1356, 1358-1362, 1381-\>1385, 1394, 1423-1437, 1453-1468, 1471-1479, 1489-1502, 1512-1527, 1549, 1562-1581, 1618-1630, 1652-1666, 1706-1707, 1716-1719, 1739-1751, 1762-1774, 1784-1840, 1860, 1872-1892, 1904-1926, 1941-1966, 1979-1981, 1998-2019, 2036-2060 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      149 |       50 |       62 |        6 |     66% |82-\>85, 111, 165-177, 182-189, 191-192, 197-209, 213-225, 230-231, 237-238, 245-248, 252-255 |
| **TOTAL**                                                                         | **1011** |  **535** |  **276** |   **32** | **46%** |           |


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