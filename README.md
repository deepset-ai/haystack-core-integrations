# Repository Coverage (weaviate)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-weaviate/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/weaviate/bm25\_retriever.py      |       32 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/embedding\_retriever.py |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/weaviate/hybrid\_retriever.py    |       47 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/\_filters.py                 |      148 |       82 |       66 |        7 |     42% |22-23, 43, 96-102, 104-105, 113-116, 122, 127-130, 136-154, 158-176, 180-198, 202-220, 224-228, 232-236, 246-249, 268-\>276, 295-296 |
| src/haystack\_integrations/document\_stores/weaviate/auth.py                      |       85 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/weaviate/document\_store.py           |      644 |      414 |      208 |       22 |     33% |189, 208, 226, 234, 253, 271, 281-283, 288-293, 299-302, 308-311, 348, 350, 352, 362-363, 369-371, 382-385, 396-400, 420-427, 447-455, 466-501, 515-551, 568-590, 606-630, 653-680, 703-732, 748-749, 753-\>763, 775, 780, 791, 797-800, 805-811, 814-821, 847-849, 855-884, 900-907, 923-930, 956-973, 982-1002, 1012-1037, 1069, 1071-\>1051, 1104-1107, 1137, 1147-1148, 1156-1158, 1175-1204, 1223-1255, 1273-1279, 1295-1313, 1333, 1354, 1361, 1374-1379, 1381-1382, 1396-1479, 1484-1495, 1500-1513, 1527-1539, 1549-1567, 1578-1592, 1603-1619 |
| **TOTAL**                                                                         | **1003** |  **496** |  **298** |   **29** | **46%** |           |


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