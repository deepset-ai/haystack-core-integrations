# Repository Coverage (weaviate)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       36 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |       82 |       66 |        7 |     42% |22-23, 43, 96-102, 104-105, 113-116, 122, 127-130, 136-154, 158-176, 180-198, 202-220, 224-228, 232-236, 246-249, 268-\>276, 295-296 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      649 |      406 |      210 |       22 |     35% |190, 209, 227, 235, 254, 272, 282-284, 289-294, 351, 353, 355, 365-366, 372-374, 385-388, 399-403, 423-430, 450-458, 469-504, 518-554, 571-593, 609-633, 655-664, 696-719, 751-776, 792-793, 797-\>807, 819, 824, 835, 841-844, 849-855, 858-865, 891-893, 899-928, 944-951, 967-974, 1000-1017, 1026-1046, 1056-1081, 1113, 1115-\>1095, 1148-1151, 1181, 1191-1192, 1200-1202, 1219-1248, 1267-1299, 1317-1323, 1339-1357, 1377, 1398, 1405, 1418-1423, 1425-1426, 1440-1523, 1528-1539, 1544-1557, 1571-1583, 1593-1611, 1622-1636, 1647-1663 |
| **TOTAL**                                                                         | **1020** |  **488** |  **300** |   **29** | **48%** |           |


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