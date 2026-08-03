# Repository Coverage (weaviate)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       36 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |       82 |       66 |        7 |     42% |22-23, 43, 96-102, 104-105, 113-116, 122, 127-130, 136-154, 158-176, 180-198, 202-220, 224-228, 232-236, 246-249, 268-\>276, 295-296 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      643 |      400 |      206 |       22 |     35% |190, 209, 227, 235, 254, 272, 282-284, 289-294, 351, 353, 355, 365-366, 372-374, 385-388, 399-403, 423-430, 450-458, 469-504, 518-554, 571-593, 609-633, 654-663, 686-701, 724-741, 757-758, 762-\>772, 784, 789, 800, 806-809, 814-820, 823-830, 856-858, 864-893, 909-916, 932-939, 965-982, 991-1011, 1021-1046, 1078, 1080-\>1060, 1113-1116, 1146, 1156-1157, 1165-1167, 1184-1213, 1232-1264, 1282-1288, 1304-1322, 1342, 1363, 1370, 1383-1388, 1390-1391, 1405-1488, 1493-1504, 1509-1522, 1536-1548, 1558-1576, 1587-1601, 1612-1628 |
| **TOTAL**                                                                         | **1014** |  **482** |  **296** |   **29** | **48%** |           |


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