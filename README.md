# Repository Coverage (optimum)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum/htmlcov/index.html)

| Name                                                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/optimum/\_backend.py                   |      167 |       85 |       44 |        5 |     46% |35-36, 55-60, 124-125, 128-133, 138-140, 153-199, 210-218, 225-228, 231-\>exit, 234-\>exit, 240-280, 284-303 |
| src/haystack\_integrations/components/embedders/optimum/optimization.py                |       38 |        4 |       10 |        2 |     88% |48-49, 83-84 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_document\_embedder.py |       53 |        8 |       10 |        1 |     83% |   225-235 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_text\_embedder.py     |       38 |        5 |        4 |        1 |     86% |   181-186 |
| src/haystack\_integrations/components/embedders/optimum/pooling.py                     |       18 |        1 |        2 |        0 |     95% |        36 |
| src/haystack\_integrations/components/embedders/optimum/quantization.py                |       38 |        4 |       10 |        2 |     88% |48-49, 83-84 |
| **TOTAL**                                                                              |  **352** |  **107** |   **80** |   **11** | **67%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-optimum/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-optimum/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-optimum%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.