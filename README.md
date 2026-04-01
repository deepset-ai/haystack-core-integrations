# Repository Coverage (optimum-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum-combined/htmlcov/index.html)

| Name                                                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/optimum/\_backend.py                   |      150 |        6 |       38 |        6 |     94% |96-101, 107-108, 196-\>exit, 199-\>exit, 232-\>235, 267 |
| src/haystack\_integrations/components/embedders/optimum/optimization.py                |       38 |       10 |       10 |        2 |     62% |48-49, 76-84 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_document\_embedder.py |       52 |        1 |       12 |        1 |     97% |       150 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_text\_embedder.py     |       37 |        1 |        6 |        1 |     95% |       130 |
| src/haystack\_integrations/components/embedders/optimum/pooling.py                     |       18 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/optimum/quantization.py                |       38 |        9 |       10 |        3 |     67% |48-49, 75, 78-84 |
| **TOTAL**                                                                              |  **333** |   **27** |   **78** |   **13** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-optimum-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-optimum-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-optimum-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.