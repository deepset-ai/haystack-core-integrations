# Repository Coverage (jina)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-jina/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/connectors/jina/reader.py                   |       57 |        2 |       10 |        2 |     94% |65-\>67, 127-128 |
| src/haystack\_integrations/components/connectors/jina/reader\_mode.py             |       15 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/jina/document\_embedder.py        |      110 |        7 |       28 |        4 |     92% |105, 131, 145-146, 177, 179, 187 |
| src/haystack\_integrations/components/embedders/jina/document\_image\_embedder.py |      113 |        3 |       26 |        2 |     96% |110, 198, 211 |
| src/haystack\_integrations/components/embedders/jina/text\_embedder.py            |       70 |        5 |       16 |        4 |     90% |90, 112, 143, 145, 152 |
| src/haystack\_integrations/components/rankers/jina/ranker.py                      |       71 |        1 |       18 |        0 |     99% |       108 |
| **TOTAL**                                                                         |  **436** |   **18** |  **100** |   **12** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-jina/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-jina/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-jina/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-jina/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-jina%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-jina/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.