# Repository Coverage (pgvector-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       48 |        4 |        6 |        3 |     87% |90-91, 94-95, 135-\>137 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       41 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        2 |       18 |        3 |     92% |31-\>41, 34, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      739 |       39 |      190 |       26 |     93% |276-\>exit, 281-\>exit, 325-\>exit, 330-\>exit, 387-388, 401-\>403, 444-\>446, 635-636, 640-641, 658-659, 677, 691-696, 714, 730-735, 763, 788, 894-\>899, 918-924, 951-\>956, 975-981, 1024, 1090-\>1094, 1112-1114, 1132-\>1136, 1154-1156, 1181-\>1186, 1204-1206, 1231-\>1236, 1254-1256, 1386-\>1390, 1481-\>1484, 1507, 1532 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      175 |        0 |       72 |        1 |     99% |   91-\>94 |
| **TOTAL**                                                                         | **1045** |   **47** |  **290** |   **34** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-pgvector-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.