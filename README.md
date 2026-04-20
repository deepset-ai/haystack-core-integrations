# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      125 |        0 |       24 |        1 |     99% | 129-\>131 |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        2 |     97% |74-\>83, 76-\>83 |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      781 |      453 |      234 |       18 |     42% |308-319, 325-336, 354-355, 371-372, 394-426, 448-481, 490-500, 511-521, 535-558, 570-593, 679-\>671, 696-\>695, 698-\>695, 717-\>714, 719-\>714, 721-\>714, 768-816, 833-880, 889-929, 940-980, 993-1005, 1017-1029, 1045-1071, 1087-1113, 1125-1150, 1162-1187, 1200-1224, 1241-1265, 1283-1311, 1329-1357, 1400-1425, 1440-1465, 1481-1498, 1512-1529, 1564-1606, 1638-1666, 1712-1716, 1722, 1776-1779, 1815-1859, 1891-1920, 1965-1969, 1975-2000, 2025, 2031-2034, 2064-2069, 2084-2089, 2243, 2286, 2310-2321, 2338-2349, 2472-2490, 2497-2500, 2519-2520, 2556-2562, 2565-2566, 2582-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |       16 |       60 |       12 |     83% |44-\>36, 48-\>36, 70, 75-80, 99, 127-128, 143, 156, 171, 189, 201, 213, 225 |
| **TOTAL**                                                             | **1073** |  **469** |  **334** |   **33** | **56%** |           |


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