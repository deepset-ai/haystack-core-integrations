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
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      832 |      372 |      270 |       33 |     54% |254-\>257, 259-\>exit, 315-\>318, 333-\>335, 337, 354-367, 398, 425-426, 435-444, 498-501, 507-510, 518, 533-\>535, 563-564, 582-584, 604-605, 607-\>610, 620-\>630, 623, 656-679, 706-\>709, 734-749, 758-764, 838-841, 848, 898-899, 948, 968-985, 1000-1016, 1032-1052, 1068-1088, 1101-1103, 1136, 1142-\>1145, 1152, 1242-1275, 1312-1364, 1391-1398, 1417-1428, 1443-1455, 1520-1560, 1625-1665, 1733-1734, 1738, 1768-1773, 1781-\>1784, 1840-1851, 1864, 1868, 1878-1883, 1893-1898, 1911-1915, 1921-1930, 1948-1954, 1968-1994, 2012-2038, 2066-2074, 2100-2108, 2115, 2132-2134, 2144-2152, 2162-2170, 2182, 2208, 2224, 2227-\>2232, 2251-2265, 2273-2287, 2359-2369, 2390-\>2397, 2404-2406, 2427-\>2434, 2441-2443 |
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