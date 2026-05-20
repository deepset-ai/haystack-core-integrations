# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       70 |        3 |       20 |        4 |     92% |173, 182-\>185, 279, 340 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       76 |        6 |       28 |       10 |     85% |250, 252, 253-\>255, 275, 370, 372, 373-\>375, 375-\>377, 377-\>380, 395 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       77 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       77 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       44 |       20 |        6 |        0 |     52% |113-130, 162-179 |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       63 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      746 |      396 |      240 |       22 |     45% |206-217, 271-\>274, 289-\>291, 293, 312-326, 339, 366-367, 376-385, 437-440, 446-449, 455-459, 464-474, 496-497, 509-511, 522-523, 525-\>528, 536-\>546, 539, 572-595, 622-\>625, 650-665, 674-680, 754-757, 764, 814-815, 864, 882-899, 912-928, 942-968, 982-1008, 1021-1023, 1056, 1062-\>1065, 1072, 1232-1284, 1311-1318, 1337-1348, 1363-1375, 1440-1480, 1545-1585, 1597-1649, 1672-1682, 1705-1716, 1729, 1733, 1743-1748, 1758-1763, 1776-1780, 1786-1795, 1813-1819, 1833-1859, 1877-1903, 1931-1939, 1965-1973, 1980, 1997-1999, 2009-2017, 2027-2035, 2058-2101, 2124-2167, 2188-\>2195, 2202-2204, 2225-\>2232, 2239-2241 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      189 |       51 |      112 |       14 |     74% |19-20, 28, 55, 138, 141, 156-165, 169-187, 196, 200-205, 207-208, 218, 222-227, 229-230, 240, 242-249, 251-252, 258-259, 264-267 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1351** |  **476** |  **440** |   **50** | **62%** |           |


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