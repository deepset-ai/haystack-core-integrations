# Repository Coverage (qdrant-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      809 |       39 |      234 |       30 |     93% |438-439, 493-494, 534-535, 576, 591-593, 611, 626-628, 731-\>730, 733-\>730, 754-\>749, 812, 832-\>818, 852-854, 877, 896-\>882, 900, 916-918, 1094-\>1109, 1106-\>1096, 1137-\>1152, 1149-\>1139, 1187-\>1177, 1225-\>1215, 1263-\>1253, 1305-\>1295, 1359-\>1349, 1415-\>1405, 2146-2147, 2165-2167, 2556, 2576, 2597-2598, 2634-2640, 2643-2644, 2660-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |        5 |       60 |        6 |     94% |44-\>36, 48-\>36, 70, 80, 99, 127-128 |
| **TOTAL**                                                             | **1115** |   **44** |  **334** |   **36** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-qdrant-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.