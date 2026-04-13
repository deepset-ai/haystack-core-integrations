# Repository Coverage (optimum)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-optimum/htmlcov/index.html)

| Name                                                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/optimum/\_backend.py                   |      153 |       66 |       38 |        5 |     55% |34-35, 104-109, 115-116, 129-172, 183-191, 198-201, 204-\>exit, 207-\>exit, 213-253, 259-261, 275 |
| src/haystack\_integrations/components/embedders/optimum/optimization.py                |       38 |       12 |       10 |        1 |     56% |48-49, 74-84 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_document\_embedder.py |       52 |       11 |       12 |        2 |     73% |149-153, 214, 227-234 |
| src/haystack\_integrations/components/embedders/optimum/optimum\_text\_embedder.py     |       37 |        8 |        6 |        2 |     72% |129-133, 174, 183-185 |
| src/haystack\_integrations/components/embedders/optimum/pooling.py                     |       18 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/optimum/quantization.py                |       38 |       12 |       10 |        1 |     56% |48-49, 74-84 |
| **TOTAL**                                                                              |  **336** |  **109** |   **78** |   **11** | **62%** |           |


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