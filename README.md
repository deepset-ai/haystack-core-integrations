# Repository Coverage (weaviate)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       36 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |       82 |       66 |        7 |     42% |22-23, 43, 96-102, 104-105, 113-116, 122, 127-130, 136-154, 158-176, 180-198, 202-220, 224-228, 232-236, 246-249, 268-\>276, 295-296 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      664 |      404 |      212 |       20 |     36% |235, 272, 280, 317, 327-329, 334-339, 396, 398, 400, 410-411, 417-419, 430-433, 444-448, 468-475, 495-503, 514-549, 563-599, 616-638, 654-678, 700-709, 741-764, 796-821, 837-838, 842-\>852, 864, 869, 880, 886-889, 894-900, 903-910, 936-938, 944-973, 989-996, 1012-1019, 1045-1062, 1071-1091, 1101-1126, 1158, 1160-\>1140, 1193-1196, 1226, 1236-1237, 1245-1247, 1264-1293, 1312-1344, 1362-1368, 1384-1402, 1422, 1443, 1450, 1463-1468, 1470-1471, 1485-1568, 1573-1584, 1589-1602, 1616-1628, 1638-1656, 1667-1681, 1692-1708 |
| **TOTAL**                                                                         | **1035** |  **486** |  **302** |   **27** | **49%** |           |


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