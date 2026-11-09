# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      809 |      250 |      234 |       30 |     66% |360, 378, 399-400, 416-417, 444-445, 448-449, 498-500, 503-504, 535-545, 580-603, 615-638, 724-\>716, 741-\>740, 743-\>740, 762-\>759, 764-\>759, 766, 772-\>759, 816-864, 881-928, 945-956, 996-1007, 1051, 1076, 1101-1119, 1144-1162, 1196-1200, 1234-1238, 1272-1276, 1314-1318, 1368-1374, 1424-1430, 1475-1500, 1515-1540, 1556-1573, 1587-1604, 1669-1681, 1732-1741, 1794-1798, 1805, 1859-1862, 1929-1942, 1993-2003, 2053-2057, 2064-2089, 2114, 2120-2123, 2153-2158, 2173-2178, 2332, 2375, 2400-2408, 2428-2436, 2588, 2607-2608, 2644-2650, 2653-2654, 2670-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      123 |       13 |       58 |       13 |     86% |48-\>40, 52-\>40, 74, 80, 86, 105, 133-134, 147, 151, 158, 167, 179, 191, 203 |
| **TOTAL**                                                             | **1113** |  **263** |  **332** |   **43** | **74%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-qdrant%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.