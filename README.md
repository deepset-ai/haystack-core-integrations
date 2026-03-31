# Repository Coverage (jina)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-jina/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/connectors/jina/reader.py                   |       46 |        4 |       10 |        4 |     86% |66->68, 124->127, 135-136, 140-141 |
| src/haystack\_integrations/components/connectors/jina/reader\_mode.py             |       15 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/jina/document\_embedder.py        |       83 |        7 |       26 |        4 |     90% |108, 134, 148-149, 185, 224, 226 |
| src/haystack\_integrations/components/embedders/jina/document\_image\_embedder.py |       86 |       11 |       22 |        4 |     84% |113, 179-188, 201, 205-206, 238 |
| src/haystack\_integrations/components/embedders/jina/text\_embedder.py            |       53 |        7 |       16 |        4 |     84% |93, 115, 128-129, 155, 157, 165 |
| src/haystack\_integrations/components/rankers/jina/ranker.py                      |       55 |        9 |       16 |        4 |     79% |66-67, 104-105, 111, 162, 174-175, 179 |
| **TOTAL**                                                                         |  **338** |   **38** |   **92** |   **20** | **86%** |           |


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