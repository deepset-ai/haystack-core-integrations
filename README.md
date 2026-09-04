# Repository Coverage (azure_documentdb-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_documentdb-combined/htmlcov/index.html)

| Name                                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/azure\_documentdb/embedding\_retriever.py  |       41 |        0 |        6 |        1 |     98% |   73-\>75 |
| src/haystack\_integrations/components/retrievers/azure\_documentdb/full\_text\_retriever.py |       41 |        2 |        6 |        1 |     94% |73-\>75, 79, 83 |
| src/haystack\_integrations/document\_stores/azure\_documentdb/document\_store.py            |      404 |       24 |      102 |       19 |     92% |153-\>155, 204-205, 220, 233, 235, 241-\>exit, 249-\>exit, 262, 354, 370, 407, 424-\>exit, 435-\>exit, 559-560, 576-578, 599-600, 616-618, 674-675, 680-681, 738-\>740, 759 |
| src/haystack\_integrations/document\_stores/azure\_documentdb/filters.py                    |       65 |        0 |       28 |        0 |    100% |           |
| **TOTAL**                                                                                   |  **551** |   **26** |  **142** |   **21** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-azure_documentdb-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_documentdb-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-azure_documentdb-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_documentdb-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-azure_documentdb-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_documentdb-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.