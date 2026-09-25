# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      809 |      312 |      234 |       21 |     58% |350, 368, 389-390, 406-407, 429-461, 483-516, 525-535, 570-593, 605-628, 714-\>706, 731-\>730, 733-\>730, 752-\>749, 754-\>749, 756, 762-\>749, 806-854, 871-918, 935-946, 986-997, 1041, 1066, 1091-1109, 1134-1152, 1186-1190, 1224-1228, 1262-1266, 1304-1308, 1358-1364, 1414-1420, 1466-1491, 1506-1531, 1547-1564, 1578-1595, 1640-1672, 1704-1732, 1784-1788, 1795, 1849-1852, 1898-1932, 1964-1993, 2042-2046, 2053-2078, 2103, 2109-2112, 2142-2147, 2162-2167, 2321, 2364, 2388-2399, 2416-2427, 2550-2568, 2575-2578, 2597-2598, 2634-2640, 2643-2644, 2660-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |       16 |       60 |       12 |     83% |44-\>36, 48-\>36, 70, 75-80, 99, 127-128, 143, 156, 171, 189, 201, 213, 225 |
| **TOTAL**                                                             | **1115** |  **328** |  **334** |   **33** | **68%** |           |


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