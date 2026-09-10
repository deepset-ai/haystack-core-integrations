# Repository Coverage (vllm-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vllm-combined/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/vllm/utils.py                               |       22 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/vllm/document\_embedder.py    |      130 |        7 |       32 |        1 |     94% |241-246, 301 |
| src/haystack\_integrations/components/embedders/vllm/text\_embedder.py        |       62 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/vllm/chat/chat\_generator.py |      173 |        4 |       52 |        5 |     96% |60, 334-\>336, 381-\>384, 426-\>429, 454-\>422, 456-458 |
| src/haystack\_integrations/components/rankers/vllm/ranker.py                  |      104 |        1 |       30 |        1 |     99% |       265 |
| **TOTAL**                                                                     |  **491** |   **12** |  **134** |    **7** | **97%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vllm-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vllm-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vllm-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vllm-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-vllm-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vllm-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.