# Repository Coverage (huggingface_api-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/huggingface\_api/utils.py                                    |       78 |        2 |       14 |        1 |     97% |   142-143 |
| src/haystack\_integrations/components/embedders/huggingface\_api/document\_embedder.py         |      212 |       12 |       76 |        9 |     93% |195-196, 327-\>331, 331-\>335, 340, 359-360, 394, 416-417, 488-492, 526-530 |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_document\_embedder.py |      162 |        4 |       48 |        3 |     97% |188, 214, 236-237 |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_embedding\_utils.py   |       42 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_text\_embedder.py     |       93 |        0 |       30 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/text\_embedder.py             |      142 |        4 |       60 |        5 |     96% |158-159, 236-\>240, 240-\>245, 360, 362 |
| src/haystack\_integrations/components/generators/huggingface\_api/chat/chat\_generator.py      |      253 |        8 |       94 |       13 |     94% |135-\>139, 137-\>139, 187, 220-\>222, 421-422, 506, 613-614, 669, 691-\>696, 724-\>717, 728-\>731, 745, 764-\>769 |
| src/haystack\_integrations/components/rankers/huggingface\_api/ranker.py                       |      132 |       10 |       42 |        3 |     93% |293-295, 335-336, 343, 348, 374-376 |
| **TOTAL**                                                                                      | **1114** |   **40** |  **378** |   **34** | **95%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-huggingface_api-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.