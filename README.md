# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/multivector\_retriever.py |       44 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py              |      130 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/utils.py                  |       17 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py                  |       51 |        0 |       22 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py             |      874 |      217 |      252 |       28 |     73% |419-420, 436-437, 464-465, 471-472, 522-524, 530-531, 563-573, 608-631, 643-666, 752-\>744, 769-\>768, 771-\>768, 790-\>787, 792-\>787, 794, 800-\>787, 844-892, 909-956, 973-987, 1027-1041, 1085, 1110, 1151-\>1141, 1195-\>1185, 1232-1236, 1270-1274, 1310-1314, 1354-1358, 1408-1414, 1464-1470, 1515-1540, 1555-1580, 1596-1613, 1627-1644, 1709-1721, 1772-1781, 1828, 1885-1889, 2027-2040, 2091-2101, 2148, 2202-2206, 2309-2314, 2329-2334, 2488, 2531, 2556-2564, 2584-2592, 2782-2783, 2842-2843, 2850-2851 |
| src/haystack\_integrations/document\_stores/qdrant/filters.py                     |      126 |       10 |       60 |       11 |     89% |48-\>40, 52-\>40, 74, 80, 86, 105, 133-134, 179, 191, 203, 215 |
| **TOTAL**                                                                         | **1242** |  **227** |  **358** |   **39** | **80%** |           |


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