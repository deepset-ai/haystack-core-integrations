# Repository Coverage (opensearch)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       70 |        3 |       20 |        4 |     92% |174, 183-\>186, 280, 341 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       76 |        6 |       28 |       10 |     85% |250, 252, 253-\>255, 275, 370, 372, 373-\>375, 375-\>377, 377-\>380, 395 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       77 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       77 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       44 |       20 |        6 |        0 |     52% |113-130, 162-179 |
| src/haystack\_integrations/components/retrievers/opensearch/utils.py                           |        8 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       63 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      743 |      395 |      240 |       22 |     45% |205-216, 270-\>273, 288-\>290, 292, 309-322, 335, 362-363, 372-381, 433-436, 442-445, 451-455, 460-470, 492-493, 505-507, 518-519, 521-\>524, 532-\>542, 535, 568-591, 618-\>621, 646-661, 670-676, 750-753, 760, 810-811, 860, 878-895, 908-924, 938-964, 978-1004, 1017-1019, 1052, 1058-\>1061, 1068, 1228-1280, 1307-1314, 1333-1344, 1359-1371, 1436-1476, 1541-1581, 1593-1645, 1668-1678, 1701-1712, 1725, 1729, 1739-1744, 1754-1759, 1772-1776, 1782-1791, 1809-1815, 1829-1855, 1873-1899, 1927-1935, 1961-1969, 1976, 1993-1995, 2005-2013, 2023-2031, 2054-2097, 2120-2163, 2184-\>2191, 2198-2200, 2221-\>2228, 2235-2237 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      189 |       51 |      112 |       14 |     74% |19-20, 28, 55, 138, 141, 156-165, 169-187, 196, 200-205, 207-208, 218, 222-227, 229-230, 240, 242-249, 251-252, 258-259, 264-267 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1348** |  **475** |  **440** |   **50** | **62%** |           |


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