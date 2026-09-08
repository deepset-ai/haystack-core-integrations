# Repository Coverage (nvidia)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-nvidia/htmlcov/index.html)

| Name                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/nvidia/document\_embedder.py    |      102 |        4 |       32 |        3 |     95% |109, 126-\>exit, 129-130, 141 |
| src/haystack\_integrations/components/embedders/nvidia/text\_embedder.py        |       76 |        4 |       24 |        3 |     93% |92, 115-\>exit, 118-119, 130 |
| src/haystack\_integrations/components/embedders/nvidia/truncate.py              |       15 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/chat/chat\_generator.py |       17 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/nvidia/generator.py            |       57 |        2 |       14 |        3 |     93% |99, 116-\>exit, 141, 193-\>196 |
| src/haystack\_integrations/components/rankers/nvidia/ranker.py                  |       96 |        2 |       36 |        1 |     98% |  125, 182 |
| src/haystack\_integrations/components/rankers/nvidia/truncate.py                |        9 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/client.py                               |       16 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/models.py                               |       29 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/utils/nvidia/nim\_backend.py                         |      120 |        0 |       32 |        4 |     97% |69-\>72, 167-\>169, 169-\>172, 190-\>192 |
| src/haystack\_integrations/utils/nvidia/utils.py                                |       46 |        0 |       24 |        0 |    100% |           |
| **TOTAL**                                                                       |  **583** |   **12** |  **170** |   **14** | **97%** |           |


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