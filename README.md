# Repository Coverage (pgvector)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        3 |       18 |        5 |     87% |31-\>41, 34, 60-\>66, 64, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      746 |      442 |      190 |       30 |     40% |276-\>exit, 281-\>exit, 322, 325-\>exit, 330-\>exit, 371, 383-392, 403-412, 422, 426-429, 433-\>435, 441-\>exit, 459, 463, 467-\>469, 475-\>exit, 483-503, 509-547, 553-597, 606-614, 624-632, 657-660, 663-664, 680-\>686, 695-725, 732-766, 777-788, 797-813, 827-850, 865-888, 894-905, 924-960, 981-1017, 1025-1036, 1049-1060, 1071-1078, 1086-1093, 1114-1142, 1162-1190, 1201-1240, 1251-1290, 1298-1316, 1334-1351, 1363-1380, 1394-1395, 1397-1401, 1425-\>1429, 1438, 1467-1481, 1497-1512, 1515-1523, 1533-1546, 1556-1571, 1593, 1606-1625, 1662-1674, 1696-1710, 1750-1751, 1760-1763, 1783-1795, 1806-1818, 1828-1884, 1904, 1916-1936, 1948-1970, 1991-2028, 2041-2043, 2072-2095, 2124-2150 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |       44 |       72 |        8 |     74% |91-\>94, 131, 203-215, 220-227, 229-230, 235-247, 251-263, 268-269, 275-276, 286, 293 |
| **TOTAL**                                                                         | **1052** |  **495** |  **290** |   **47** | **52%** |           |


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