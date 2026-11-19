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
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      832 |      372 |      270 |       33 |     54% |254-\>257, 259-\>exit, 315-\>318, 333-\>335, 337, 354-367, 398, 425-426, 435-444, 498-501, 507-510, 518, 533-\>535, 557-558, 570-572, 592-593, 595-\>598, 608-\>618, 611, 644-667, 694-\>697, 722-737, 746-752, 826-829, 836, 886-887, 936, 954-971, 984-1000, 1014-1034, 1048-1068, 1081-1083, 1116, 1122-\>1125, 1132, 1222-1255, 1292-1344, 1371-1378, 1397-1408, 1423-1435, 1500-1540, 1605-1645, 1713-1714, 1718, 1748-1753, 1761-\>1764, 1820-1831, 1844, 1848, 1858-1863, 1873-1878, 1891-1895, 1901-1910, 1928-1934, 1948-1974, 1992-2018, 2046-2054, 2080-2088, 2095, 2112-2114, 2124-2132, 2142-2150, 2162, 2188, 2204, 2207-\>2212, 2231-2245, 2253-2267, 2339-2349, 2370-\>2377, 2384-2386, 2407-\>2414, 2421-2423 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      195 |       45 |      114 |       15 |     78% |19-20, 26-\>28, 55, 141, 162-171, 180, 182-189, 191-192, 202, 206-211, 213-214, 224, 228-233, 235-236, 246, 248-255, 257-258, 264-265, 270-273 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        2 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1521** |  **446** |  **482** |   **62** | **69%** |           |


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