# Repository Coverage (weaviate)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       32 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |      116 |       66 |        1 |     17% |21-23, 37-43, 70, 87-108, 112-117, 121-123, 127-130, 136-154, 158-176, 180-198, 202-220, 224-228, 232-236, 246-249, 266-285, 295-296 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        2 |        6 |        1 |     97% |     71-72 |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      634 |      524 |      200 |        3 |     14% |188-228, 233-273, 278-283, 288-293, 299-302, 308-311, 346, 348, 350, 360-361, 367-369, 380-383, 394-398, 418-425, 445-453, 464-497, 511-545, 562-582, 598-619, 642-669, 692-721, 728-752, 759-791, 794-800, 803-810, 813-841, 844-873, 889-896, 912-919, 924-935, 945-962, 971-991, 1001-1026, 1036-1064, 1091-1094, 1123-1126, 1134-1135, 1143-1145, 1162-1191, 1210-1242, 1255-1272, 1282-1300, 1311-1372, 1383-1466, 1471-1482, 1487-1500, 1510-1526, 1536-1554, 1565-1579, 1590-1606 |
| **TOTAL**                                                                         |  **993** |  **642** |  **290** |    **5** | **30%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-weaviate/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-weaviate/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-weaviate%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.