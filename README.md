# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      139 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      806 |      328 |      234 |       21 |     56% |348, 365, 388-389, 405-406, 428-460, 482-515, 524-534, 569-592, 604-627, 713-\>705, 730-\>729, 732-\>729, 751-\>748, 753-\>748, 755, 761-\>748, 805-853, 870-917, 926-966, 977-1017, 1039, 1063, 1087-1105, 1129-1147, 1180-1184, 1217-1221, 1254-1258, 1295-1299, 1348-1354, 1403-1409, 1455-1480, 1495-1520, 1536-1553, 1567-1584, 1629-1661, 1693-1721, 1773-1777, 1784, 1838-1841, 1887-1921, 1953-1982, 2031-2035, 2042-2067, 2092, 2098-2101, 2131-2136, 2151-2156, 2310, 2353, 2377-2388, 2405-2416, 2539-2557, 2564-2567, 2586-2587, 2623-2629, 2632-2633, 2649-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |       16 |       60 |       12 |     83% |44-\>36, 48-\>36, 70, 75-80, 99, 127-128, 143, 156, 171, 189, 201, 213, 225 |
| **TOTAL**                                                             | **1112** |  **344** |  **334** |   **33** | **66%** |           |


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