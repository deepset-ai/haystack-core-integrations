# Repository Coverage (nvidia-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia-combined/htmlcov/index.html)

| Name                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/nvidia/document\_embedder.py    |      101 |        5 |       32 |        4 |     93% |110, 127-\>exit, 130-131, 138, 142 |
| src/haystack\_integrations/components/embedders/nvidia/text\_embedder.py        |       75 |        5 |       24 |        4 |     91% |93, 116-\>exit, 119-120, 127, 131 |
| src/haystack\_integrations/components/embedders/nvidia/truncate.py              |       15 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/chat/chat\_generator.py |       17 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/generator.py            |       56 |        2 |       14 |        2 |     94% |93, 110-\>exit, 135 |
| src/haystack\_integrations/components/rankers/nvidia/ranker.py                  |       95 |        3 |       36 |        1 |     95% |124, 179-180 |
| src/haystack\_integrations/components/rankers/nvidia/truncate.py                |        9 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/client.py                               |       16 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/models.py                               |       29 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/nim\_backend.py                         |      109 |        0 |       30 |        3 |     98% |147-\>149, 149-\>152, 170-\>172 |
| src/haystack\_integrations/utils/nvidia/utils.py                                |       46 |        0 |       24 |        0 |    100% |           |
| **TOTAL**                                                                       |  **568** |   **15** |  **168** |   **14** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-nvidia-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-nvidia-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-nvidia-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.