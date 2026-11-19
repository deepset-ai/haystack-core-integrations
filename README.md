# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       74 |        3 |       20 |        4 |     93% |186, 195-\>198, 292, 353 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       80 |        6 |       28 |       10 |     85% |262, 264, 265-\>267, 287, 382, 384, 385-\>387, 387-\>389, 389-\>392, 407 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       81 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/multivector\_retriever.py          |       59 |        0 |       10 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       81 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       48 |       20 |        6 |        0 |     56% |125-142, 174-191 |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       61 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      832 |      372 |      270 |       33 |     54% |254-\>257, 259-\>exit, 315-\>318, 333-\>335, 337, 354-367, 398, 425-426, 435-444, 498-501, 507-510, 518, 533-\>535, 561-562, 578-580, 600-601, 603-\>606, 616-\>626, 619, 652-675, 702-\>705, 730-745, 754-760, 834-837, 844, 894-895, 944, 964-981, 996-1012, 1028-1048, 1064-1084, 1097-1099, 1132, 1138-\>1141, 1148, 1238-1271, 1308-1360, 1387-1394, 1413-1424, 1439-1451, 1516-1556, 1621-1661, 1729-1730, 1734, 1764-1769, 1777-\>1780, 1836-1847, 1860, 1864, 1874-1879, 1889-1894, 1907-1911, 1917-1926, 1944-1950, 1964-1990, 2008-2034, 2062-2070, 2096-2104, 2111, 2128-2130, 2140-2148, 2158-2166, 2178, 2204, 2220, 2223-\>2228, 2247-2261, 2269-2283, 2355-2365, 2386-\>2393, 2400-2402, 2423-\>2430, 2437-2439 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      201 |       36 |      118 |       16 |     84% |19-20, 26-\>28, 55, 148, 185, 187-194, 196-197, 207, 211-216, 218-219, 229, 233-238, 240-241, 251, 253-260, 262-263, 269-270, 279-280 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        2 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1527** |  **437** |  **486** |   **63** | **70%** |           |


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