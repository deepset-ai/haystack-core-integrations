# Repository Coverage (opensearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       74 |        3 |       20 |        4 |     93% |186, 195-\>198, 292, 353 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       80 |        6 |       28 |       10 |     85% |262, 264, 265-\>267, 287, 382, 384, 385-\>387, 387-\>389, 389-\>392, 407 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       81 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       81 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       48 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       61 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      784 |       62 |      252 |       34 |     90% |216-\>218, 218-\>220, 220-\>223, 223-\>exit, 278-\>281, 296-\>298, 300, 361, 388-389, 398-407, 479, 558-\>568, 561, 644-\>647, 676-679, 696-702, 919-921, 948-950, 982-984, 1016-1018, 1031-1033, 1082, 1203, 1322-1324, 1352, 1354-\>1353, 1462, 1481-1483, 1486-\>1490, 1567, 1586-1588, 1591-\>1595, 1608-1609, 1788-\>1787, 1825-\>1824, 1827-\>1824, 1856-1857, 1862, 1900-1901, 1906, 2083, 2135, 2157 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      195 |        5 |      114 |        5 |     97% |19-20, 26-\>28, 55, 166, 169 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1413** |   **76** |  **454** |   **53** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-opensearch-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.