# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      125 |       28 |       24 |        6 |     72% |77-78, 129-\>131, 163-186, 218-241, 305-306, 396-419, 456-479, 548-549, 637, 697 |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |       17 |       16 |        3 |     52% |22-48, 71-\>83, 74-\>83, 76-\>83 |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      781 |      543 |      234 |       14 |     27% |49-53, 308-319, 325-336, 354-355, 371-372, 394-426, 448-481, 490-500, 511-521, 535-558, 570-593, 603, 621-630, 640-645, 655-664, 671-681, 692-702, 714-726, 739-747, 768-816, 833-880, 889-929, 940-980, 993-1005, 1017-1029, 1045-1071, 1087-1113, 1125-1150, 1162-1187, 1200-1224, 1241-1265, 1283-1311, 1329-1357, 1400-1425, 1440-1465, 1481-1498, 1512-1529, 1564-1606, 1638-1666, 1712-1716, 1722, 1776-1779, 1815-1859, 1891-1920, 1965-1969, 1975-2000, 2025, 2031-2034, 2049-2055, 2064-2069, 2084-2089, 2243, 2286, 2310-2321, 2338-2349, 2356-2370, 2424, 2427, 2458-2464, 2472-2490, 2497-2500, 2519-2520, 2556-2562, 2565-2566, 2582-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |      102 |       60 |        2 |     14% |20, 24-29, 34-54, 59-80, 85-93, 98-109, 114-130, 134-136, 140-143, 156, 168-171, 184-192, 196-204, 208-216, 220-228, 233-237 |
| **TOTAL**                                                             | **1073** |  **690** |  **334** |   **25** | **31%** |           |


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