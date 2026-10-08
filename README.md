# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       74 |        3 |       20 |        4 |     93% |186, 195-\>198, 292, 353 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       80 |        6 |       28 |       10 |     85% |262, 264, 265-\>267, 287, 382, 384, 385-\>387, 387-\>389, 389-\>392, 407 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       81 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       81 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       48 |       20 |        6 |        0 |     56% |125-142, 174-191 |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       61 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      784 |      406 |      252 |       26 |     46% |213-224, 278-\>281, 296-\>298, 300, 317-330, 361, 388-389, 398-407, 459-462, 468-471, 477-481, 486-496, 504-506, 518-519, 531-533, 544-545, 547-\>550, 558-\>568, 561, 594-617, 644-\>647, 672-687, 696-702, 776-779, 786, 836-837, 886, 904-921, 934-950, 964-984, 998-1018, 1031-1033, 1066, 1072-\>1075, 1082, 1172-1205, 1242-1294, 1321-1328, 1347-1358, 1373-1385, 1450-1490, 1555-1595, 1607-1659, 1682-1692, 1715-1726, 1739, 1743, 1753-1758, 1768-1773, 1786-1790, 1796-1805, 1823-1829, 1843-1869, 1887-1913, 1941-1949, 1975-1983, 1990, 2007-2009, 2019-2027, 2037-2045, 2057, 2083, 2099, 2102-\>2107, 2126-2140, 2148-2162, 2234-2244, 2265-\>2272, 2279-2281, 2302-\>2309, 2316-2318 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      195 |       45 |      114 |       15 |     78% |19-20, 26-\>28, 55, 141, 162-171, 180, 182-189, 191-192, 202, 206-211, 213-214, 224, 228-233, 235-236, 246, 248-255, 257-258, 264-265, 270-273 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1413** |  **480** |  **454** |   **55** | **64%** |           |


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