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
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      734 |      437 |      236 |       21 |     39% |206-217, 271-\>274, 289-\>291, 293, 312-326, 339, 366-367, 376-385, 394-408, 422-425, 431-434, 440-443, 449-453, 458-468, 490-491, 503-505, 516-517, 519-\>522, 530-\>540, 533, 566-589, 616-\>619, 644-659, 668-674, 748-751, 758, 775-807, 819-844, 858-875, 888-904, 918-944, 958-984, 997-999, 1032, 1038-\>1041, 1048, 1208-1260, 1287-1294, 1313-1324, 1339-1351, 1416-1456, 1521-1561, 1573-1625, 1648-1658, 1681-1692, 1705, 1709, 1719-1724, 1734-1739, 1752-1756, 1762-1771, 1789-1795, 1809-1834, 1852-1877, 1905-1912, 1938-1945, 1952, 1969-1971, 1981-1989, 1999-2007, 2030-2073, 2096-2139, 2160-\>2167, 2174-2176, 2197-\>2204, 2211-2213 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      189 |       51 |      112 |       14 |     74% |19-20, 28, 55, 138, 141, 156-165, 169-187, 196, 200-205, 207-208, 218, 222-227, 229-230, 240, 242-249, 251-252, 258-259, 264-267 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1339** |  **517** |  **436** |   **49** | **59%** |           |


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