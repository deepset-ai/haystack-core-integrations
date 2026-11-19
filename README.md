# Repository Coverage (opensearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       74 |        3 |       20 |        4 |     93% |186, 195-\>198, 292, 353 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       80 |        6 |       28 |       10 |     85% |262, 264, 265-\>267, 287, 382, 384, 385-\>387, 387-\>389, 389-\>392, 407 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       81 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/multivector\_retriever.py          |       59 |        0 |       10 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       81 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       48 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       61 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      832 |       62 |      270 |       32 |     91% |254-\>257, 259-\>exit, 315-\>318, 333-\>335, 337, 398, 425-426, 435-444, 518, 616-\>626, 619, 702-\>705, 734-737, 754-760, 979-981, 1010-1012, 1046-1048, 1082-1084, 1097-1099, 1148, 1269, 1388-1390, 1418, 1420-\>1419, 1528, 1547-1549, 1552-\>1556, 1633, 1652-1654, 1657-\>1661, 1729-1730, 1909-\>1908, 1946-\>1945, 1948-\>1945, 1977-1978, 1983, 2021-2022, 2027, 2204, 2256, 2278 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      201 |        3 |      118 |        3 |     98% |19-20, 26-\>28, 55 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        2 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1527** |   **74** |  **486** |   **49** | **93%** |           |


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