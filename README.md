# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       44 |        4 |        6 |        3 |     86% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       37 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      716 |      476 |      182 |       17 |     33% |253-\>exit, 258-\>exit, 299, 302-\>exit, 307-\>exit, 348, 357-387, 398-430, 437-457, 463-501, 507-551, 560-568, 578-586, 595-638, 647-677, 684-718, 729-740, 749-765, 779-802, 817-840, 846-857, 876-912, 933-964, 972-985, 995-1008, 1018-1025, 1033-1040, 1054-1086, 1096-1128, 1139-1178, 1189-1228, 1236-1254, 1272-1289, 1301-1318, 1332-1333, 1335-1339, 1358-\>1362, 1371, 1400-1414, 1430-1445, 1448-1456, 1466-1479, 1489-1504, 1526, 1539-1558, 1595-1607, 1629-1643, 1683-1684, 1693-1696, 1716-1728, 1739-1751, 1761-1817, 1837, 1849-1869, 1881-1903, 1917-1944, 1957-1959, 1976-1997, 2014-2038 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      149 |       50 |       62 |        6 |     66% |82-\>85, 111, 165-177, 182-189, 191-192, 197-209, 213-225, 230-231, 237-238, 245-248, 252-255 |
| **TOTAL**                                                                         |  **988** |  **535** |  **272** |   **32** | **45%** |           |


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