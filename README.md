# Repository Coverage (ollama)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-ollama/htmlcov/index.html)

| Name                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/ollama/document\_embedder.py    |       75 |       30 |       16 |        1 |     55% |106-113, 119-135, 186-187, 209-224, 244-259 |
| src/haystack\_integrations/components/embedders/ollama/text\_embedder.py        |       22 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/ollama/chat/chat\_generator.py |      231 |        5 |       72 |        9 |     95% |62, 143-\>145, 145-\>147, 147-\>155, 409-\>413, 477-478, 491-\>495, 498, 686 |
| src/haystack\_integrations/components/generators/ollama/generator.py            |       71 |       35 |       12 |        1 |     45% |63-75, 187-\>189, 195-199, 206-211, 219-225, 231-236, 261-278 |
| **TOTAL**                                                                       |  **399** |   **70** |  **100** |   **11** | **80%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-ollama/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-ollama/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-ollama/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-ollama/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-ollama%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-ollama/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.