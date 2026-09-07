# Repository Coverage (huggingface_api)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/huggingface\_api/utils.py                                    |       65 |        5 |       10 |        1 |     89% |   137-142 |
| src/haystack\_integrations/components/embedders/huggingface\_api/document\_embedder.py         |      152 |        6 |       46 |        5 |     94% |179-180, 288-\>292, 292-\>296, 389-393, 425-429 |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_document\_embedder.py |       83 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_embedding\_utils.py   |       42 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/sparse\_text\_embedder.py     |       41 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/huggingface\_api/text\_embedder.py             |      108 |        4 |       40 |        6 |     93% |143-144, 198-\>208, 199-\>203, 203-\>208, 304, 306 |
| src/haystack\_integrations/components/generators/huggingface\_api/chat/chat\_generator.py      |      260 |        8 |       96 |       14 |     94% |136-\>140, 138-\>140, 188, 221-\>223, 422-423, 519, 626-627, 665-\>668, 682, 704-\>709, 737-\>730, 741-\>744, 758, 777-\>782 |
| src/haystack\_integrations/components/rankers/huggingface\_api/ranker.py                       |       91 |        9 |       24 |        2 |     90% |234-236, 276-277, 284, 301-303 |
| **TOTAL**                                                                                      |  **842** |   **32** |  **246** |   **28** | **94%** |           |


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