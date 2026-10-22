# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      809 |      312 |      234 |       21 |     58% |360, 378, 399-400, 416-417, 439-471, 493-526, 535-545, 580-603, 615-638, 724-\>716, 741-\>740, 743-\>740, 762-\>759, 764-\>759, 766, 772-\>759, 816-864, 881-928, 945-956, 996-1007, 1051, 1076, 1101-1119, 1144-1162, 1196-1200, 1234-1238, 1272-1276, 1314-1318, 1368-1374, 1424-1430, 1475-1500, 1515-1540, 1556-1573, 1587-1604, 1649-1681, 1713-1741, 1794-1798, 1805, 1859-1862, 1908-1942, 1974-2003, 2053-2057, 2064-2089, 2114, 2120-2123, 2153-2158, 2173-2178, 2332, 2375, 2399-2410, 2427-2438, 2562-2580, 2587-2590, 2609-2610, 2646-2652, 2655-2656, 2672-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      123 |       16 |       58 |       12 |     82% |48-\>40, 52-\>40, 74, 79-84, 103, 131-132, 145, 149, 156, 165, 177, 189, 201 |
| **TOTAL**                                                             | **1113** |  **328** |  **332** |   **33** | **68%** |           |


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