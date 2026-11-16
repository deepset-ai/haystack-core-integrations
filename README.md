# Repository Coverage (azure_ai_search)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-azure_ai_search/htmlcov/index.html)

| Name                                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/bm25\_retriever.py      |       40 |        0 |        6 |        1 |     98% |   95-\>97 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/embedding\_retriever.py |       40 |        0 |        6 |        1 |     98% |   92-\>94 |
| src/haystack\_integrations/components/retrievers/azure\_ai\_search/hybrid\_retriever.py    |       40 |        0 |        6 |        1 |     98% |   95-\>97 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/document\_store.py           |      386 |       92 |      136 |       18 |     75% |71-\>76, 214, 228-235, 243-244, 294-318, 413, 421-422, 431-\>434, 440-443, 517-526, 536-540, 551-552, 565-569, 630-\>635, 632-633, 636, 644-\>647, 673-677, 686-715, 728-743, 763-764, 771, 791, 803, 814-815, 852, 857, 914-\>917, 950-953, 983-985, 1021-1030 |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/errors.py                    |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/azure\_ai\_search/filters.py                   |       77 |        0 |       36 |        0 |    100% |           |
| **TOTAL**                                                                                  |  **591** |   **92** |  **190** |   **21** | **83%** |           |


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