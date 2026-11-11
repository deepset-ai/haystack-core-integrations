# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/multivector\_retriever.py |       44 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py              |      129 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/qdrant/utils.py                  |       17 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py                  |       51 |        0 |       22 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py             |      869 |      228 |      252 |       31 |     72% |417-418, 434-435, 462-463, 469-470, 520-522, 528-529, 561-571, 606-629, 641-664, 750-\>742, 767-\>766, 769-\>766, 788-\>785, 790-\>785, 792, 798-\>785, 842-890, 907-954, 971-985, 1025-1039, 1083, 1108, 1149-\>1139, 1193-\>1183, 1230-1234, 1268-1272, 1306-1310, 1348-1352, 1402-1408, 1458-1464, 1509-1534, 1549-1574, 1590-1607, 1621-1638, 1703-1715, 1766-1775, 1822, 1875-1879, 1886, 1940-1943, 2010-2023, 2074-2084, 2131, 2181-2185, 2192-2217, 2242, 2248-2251, 2281-2286, 2301-2306, 2460, 2503, 2528-2536, 2556-2564, 2735, 2754-2755, 2814-2815, 2822-2823 |
| src/haystack\_integrations/document\_stores/qdrant/filters.py                     |      126 |       10 |       60 |       11 |     89% |48-\>40, 52-\>40, 74, 80, 86, 105, 133-134, 179, 191, 203, 215 |
| **TOTAL**                                                                         | **1236** |  **238** |  **358** |   **42** | **79%** |           |


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