# Repository Coverage (azure_ai_search)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

| Name                                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/bm25\_retriever.py      |       39 |        5 |        6 |        2 |     84% |59-60, 95->97, 131-136 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/embedding\_retriever.py |       39 |        5 |        6 |        2 |     84% |56-57, 92->94, 124-129 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/hybrid\_retriever.py    |       39 |        5 |        6 |        2 |     84% |59-60, 95->97, 133-138 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/document\_store.py           |      354 |      187 |      122 |       13 |     44% |195, 209-216, 224-225, 245-246, 249-254, 258-259, 275-299, 383, 391-392, 401->404, 406-407, 431, 432->428, 441-450, 457-471, 481-490, 500-504, 515-516, 529-533, 553->557, 580-595, 603-607, 616-645, 658-673, 687-721, 730, 741-742, 754-759, 765-794, 807-808, 817-824, 829-836, 865-872, 898-904, 933-949 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/errors.py                    |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/filters.py                   |       71 |       46 |       32 |        5 |     29% |15-17, 21, 25-48, 54-56, 64-65, 73-75, 79-83, 87-91, 95-96, 101-109 |
| **TOTAL**                                                                                  |  **550** |  **248** |  **172** |   **24** | **50%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-azure_ai_search/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-azure_ai_search/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-azure_ai_search%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.