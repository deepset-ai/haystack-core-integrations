# Repository Coverage (nvidia)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia/htmlcov/index.html)

| Name                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/nvidia/document\_embedder.py    |      101 |        6 |       32 |        5 |     92% |110, 127-\>exit, 130-131, 138, 142, 254 |
| src/haystack\_integrations/components/embedders/nvidia/text\_embedder.py        |       75 |        6 |       24 |        5 |     89% |93, 116-\>exit, 119-120, 127, 131, 206 |
| src/haystack\_integrations/components/embedders/nvidia/truncate.py              |       15 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/chat/chat\_generator.py |       17 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/generator.py            |       56 |       23 |       14 |        2 |     53% |93, 97-114, 121, 134-137, 159, 171-173, 187-193 |
| src/haystack\_integrations/components/rankers/nvidia/ranker.py                  |       95 |        5 |       36 |        3 |     92% |124, 179-180, 209, 225 |
| src/haystack\_integrations/components/rankers/nvidia/truncate.py                |        9 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/client.py                               |       16 |        2 |        2 |        1 |     83% |     28-29 |
| src/haystack\_integrations/utils/nvidia/models.py                               |       29 |        9 |        4 |        0 |     61% | 36, 40-53 |
| src/haystack\_integrations/utils/nvidia/nim\_backend.py                         |      109 |       18 |       30 |        5 |     83% |91-94, 125-128, 147-\>149, 149-\>152, 170-\>172, 175-177, 196-199, 203-205 |
| src/haystack\_integrations/utils/nvidia/utils.py                                |       46 |        0 |       24 |        0 |    100% |           |
| **TOTAL**                                                                       |  **568** |   **69** |  **168** |   **21** | **86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-nvidia/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-nvidia/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-nvidia%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.