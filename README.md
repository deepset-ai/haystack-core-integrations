# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      751 |      444 |      190 |       30 |     40% |277-\>exit, 282-\>exit, 323, 326-\>exit, 331-\>exit, 372, 384-393, 404-413, 423, 427-430, 434-\>436, 442-\>exit, 460, 464, 468-\>470, 476-\>exit, 484-504, 510-548, 554-598, 607-615, 625-633, 717, 802, 828-831, 834-835, 851-\>857, 866-896, 903-937, 948-959, 968-984, 998-1021, 1036-1059, 1065-1076, 1095-1131, 1152-1188, 1196-1207, 1220-1231, 1242-1249, 1257-1264, 1285-1313, 1333-1361, 1372-1411, 1422-1461, 1469-1487, 1505-1522, 1534-1551, 1565-1566, 1568-1572, 1596-\>1600, 1609, 1638-1652, 1668-1683, 1686-1694, 1704-1717, 1727-1742, 1764, 1777-1796, 1833-1845, 1867-1881, 1921-1922, 1931-1934, 1954-1966, 1977-1989, 1999-2055, 2075, 2087-2107, 2119-2141, 2162-2199, 2212-2214, 2243-2266, 2295-2321 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |       44 |       72 |        8 |     74% |91-\>94, 131, 203-215, 220-227, 229-230, 235-247, 251-263, 268-269, 275-276, 286, 293 |
| src/haystack\_integrations/document\_stores/pgvector/table\_replacement.py        |      219 |       64 |       68 |        0 |     74% |196, 200-201, 242, 245, 255, 263-264, 325-329, 336-343, 364-388, 409-440 |
| **TOTAL**                                                                         | **1276** |  **561** |  **358** |   **47** | **56%** |           |


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