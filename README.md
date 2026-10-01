# Repository Coverage (huggingface_api)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/huggingface\_api/utils.py                                    |       78 |        5 |       14 |        1 |     91% |   138-143 |
| src/haystack\_integrations/components/embedders/huggingface\_api/document\_embedder.py         |      212 |       14 |       76 |       11 |     91% |195-196, 327-\>331, 331-\>335, 340, 359-360, 394, 416-417, 488-492, 497, 526-530, 535 |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_document\_embedder.py |      162 |        4 |       48 |        3 |     97% |188, 214, 236-237 |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_embedding\_utils.py   |       42 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_text\_embedder.py     |       93 |        0 |       30 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/text\_embedder.py             |      142 |        4 |       60 |        5 |     96% |158-159, 236-\>240, 240-\>245, 360, 362 |
| src/haystack\_integrations/components/generators/huggingface\_api/chat/chat\_generator.py      |      260 |        8 |       96 |       14 |     94% |136-\>140, 138-\>140, 188, 221-\>223, 422-423, 519, 626-627, 665-\>668, 682, 704-\>709, 737-\>730, 741-\>744, 758, 777-\>782 |
| src/haystack\_integrations/components/rankers/huggingface\_api/ranker.py                       |      132 |       10 |       42 |        3 |     93% |293-295, 335-336, 343, 348, 374-376 |
| **TOTAL**                                                                                      | **1121** |   **45** |  **380** |   **37** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-huggingface_api%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.