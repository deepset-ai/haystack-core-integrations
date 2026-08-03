# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      804 |      326 |      236 |       21 |     56% |348, 365, 388-389, 405-406, 428-460, 482-515, 524-534, 569-592, 604-627, 713-\>705, 730-\>729, 732-\>729, 750-\>747, 752-\>747, 754, 756-\>747, 800-848, 865-912, 921-961, 972-1012, 1034, 1058, 1082-1100, 1124-1142, 1175-1179, 1212-1216, 1249-1253, 1290-1294, 1343-1348, 1397-1402, 1448-1473, 1488-1513, 1529-1546, 1560-1577, 1622-1654, 1686-1714, 1766-1770, 1777, 1831-1834, 1880-1914, 1946-1975, 2024-2028, 2035-2060, 2085, 2091-2094, 2124-2129, 2144-2149, 2303, 2346, 2370-2381, 2398-2409, 2532-2550, 2557-2560, 2579-2580, 2616-2622, 2625-2626, 2642-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |       16 |       60 |       12 |     83% |44-\>36, 48-\>36, 70, 75-80, 99, 127-128, 143, 156, 171, 189, 201, 213, 225 |
| **TOTAL**                                                             | **1110** |  **342** |  **336** |   **33** | **66%** |           |


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