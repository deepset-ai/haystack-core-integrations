# Repository Coverage (azure_ai_search)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

| Name                                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/bm25\_retriever.py      |       39 |        0 |        6 |        1 |     98% |   95-\>97 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/embedding\_retriever.py |       39 |        0 |        6 |        1 |     98% |   92-\>94 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/hybrid\_retriever.py    |       39 |        0 |        6 |        1 |     98% |   95-\>97 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/document\_store.py           |      362 |      137 |      126 |       11 |     61% |68-\>73, 210, 224-231, 239-240, 290-314, 396, 404-405, 414-\>417, 419-420, 494-503, 513-517, 528-529, 542-546, 566-\>570, 593-608, 616-620, 629-658, 671-686, 700-734, 743, 754-755, 767-772, 778-807, 842-849, 882-885, 915-917, 953-962 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/errors.py                    |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/filters.py                   |       71 |        0 |       32 |        0 |    100% |           |
| **TOTAL**                                                                                  |  **558** |  **137** |  **176** |   **14** | **74%** |           |


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