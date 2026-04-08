# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       79 |       12 |       28 |        6 |     81% |102-103, 171, 180-\>183, 273-274, 283, 340-341, 348-351 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       85 |       15 |       36 |       12 |     76% |110-111, 248, 250, 251-\>253, 262-263, 279, 374, 376, 377-\>379, 379-\>381, 381-\>384, 388-389, 403-406 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       82 |        4 |       16 |        2 |     94% |257-258, 371-372 |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       77 |        0 |       14 |        3 |     97% |346-\>350, 350-\>354, 354-\>358 |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       53 |       30 |       14 |        0 |     37% |111-134, 166-189 |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       63 |        7 |        4 |        0 |     90% |66-69, 168-173 |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      676 |      435 |      206 |       17 |     33% |191-202, 255-\>258, 273-\>275, 277, 296-310, 313-324, 335-338, 344-347, 353-356, 362-366, 371-379, 401-402, 414-416, 427-428, 430-\>433, 441-\>451, 444, 477-500, 549-557, 566-572, 646-649, 656, 673-705, 717-742, 756-773, 786-802, 816-842, 856-882, 895-897, 928, 934-\>937, 944, 1104-1156, 1183-1190, 1209-1220, 1235-1247, 1312-1352, 1417-1457, 1469-1515, 1538-1548, 1571-1582, 1595, 1599, 1609-1614, 1624-1629, 1642-1646, 1653-1662, 1680-1686, 1700-1725, 1743-1768, 1796-1803, 1829-1836, 1843, 1860-1862, 1872-1880, 1890-1898, 1921-1964, 1987-2030, 2046-2063, 2079-2096 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      135 |       50 |       72 |       12 |     63% |15-16, 19, 48, 51, 66-75, 79-97, 106, 110-115, 117-118, 128, 132-137, 139-140, 150, 152-159, 161-162, 168-169, 174-177 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1251** |  **553** |  **390** |   **52** | **53%** |           |


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