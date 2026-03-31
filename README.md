# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       79 |       12 |       28 |        6 |     81% |102-103, 171, 180->183, 273-274, 283, 340-341, 348-351 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       85 |       15 |       36 |       12 |     76% |110-111, 248, 250, 251->253, 262-263, 279, 374, 376, 377->379, 379->381, 381->384, 388-389, 403-406 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       82 |        4 |       16 |        2 |     94% |257-258, 371-372 |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       77 |        0 |       14 |        3 |     97% |346->350, 350->354, 354->358 |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       53 |       30 |       14 |        0 |     37% |111-134, 166-189 |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       63 |        7 |        4 |        0 |     90% |66-69, 168-173 |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      664 |      476 |      206 |       10 |     26% |190-201, 254->257, 272->274, 276, 280->exit, 294-309, 312-323, 334-337, 343-346, 352-355, 359-367, 370-378, 381-383, 386-388, 400-401, 413-415, 426-427, 429->432, 440->450, 443, 476-499, 548-556, 565-571, 645-648, 655, 672-704, 716-741, 755-772, 785-801, 815-841, 855-881, 893-936, 940-946, 972-984, 1011-1024, 1061-1113, 1140-1147, 1166-1177, 1192-1204, 1269-1309, 1374-1414, 1426-1472, 1495-1505, 1528-1539, 1549-1556, 1566-1571, 1581-1586, 1599-1603, 1610-1619, 1637-1643, 1657-1682, 1700-1725, 1753-1760, 1786-1793, 1800, 1817-1819, 1829-1837, 1847-1855, 1878-1921, 1944-1987, 2003-2020, 2036-2053 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      135 |       50 |       72 |       12 |     63% |15-16, 19, 48, 51, 66-75, 79-97, 106, 110-115, 117-118, 128, 132-137, 139-140, 150, 152-159, 161-162, 168-169, 174-177 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1239** |  **594** |  **390** |   **45** | **49%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-opensearch%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.