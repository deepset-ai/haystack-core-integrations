# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      807 |      230 |      230 |       32 |     69% |360, 378, 399-400, 416-417, 444-445, 448-449, 498-500, 503-504, 535-545, 580-603, 615-638, 724-\>716, 741-\>740, 743-\>740, 762-\>759, 764-\>759, 766, 772-\>759, 816-864, 881-928, 945-956, 996-1007, 1051, 1076, 1117-\>1107, 1161-\>1151, 1198-1202, 1236-1240, 1274-1278, 1316-1320, 1370-1376, 1426-1432, 1477-1502, 1517-1542, 1558-1575, 1589-1606, 1671-1683, 1734-1743, 1796-1800, 1807, 1861-1864, 1931-1944, 1995-2005, 2055-2059, 2066-2091, 2116, 2122-2125, 2155-2160, 2175-2180, 2334, 2377, 2402-2410, 2430-2438, 2590, 2609-2610, 2646-2652, 2655-2656, 2672-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      126 |       10 |       60 |       11 |     89% |48-\>40, 52-\>40, 74, 80, 86, 105, 133-134, 179, 191, 203, 215 |
| **TOTAL**                                                             | **1114** |  **240** |  **330** |   **43** | **76%** |           |


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