# Repository Coverage (cohere)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/cohere/document\_embedder.py        |       77 |        1 |       16 |        1 |     98% |       240 |
| src/haystack\_integrations/components/embedders/cohere/document\_image\_embedder.py |      111 |        0 |       26 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/cohere/embedding\_types.py          |       17 |        3 |        2 |        1 |     79% | 25, 35-36 |
| src/haystack\_integrations/components/embedders/cohere/text\_embedder.py            |       53 |        9 |        6 |        1 |     83% |105-\>exit, 167-179, 200-213 |
| src/haystack\_integrations/components/embedders/cohere/utils.py                     |       29 |       23 |       14 |        0 |     14% |36-57, 88-115 |
| src/haystack\_integrations/components/generators/cohere/chat/chat\_generator.py     |      265 |       61 |      122 |       25 |     72% |59, 79-82, 94, 102-109, 133-\>130, 141-154, 174-\>180, 176-\>175, 182-189, 193-\>195, 203, 245-249, 252-\>255, 259-\>317, 262-\>317, 265-\>317, 269-\>317, 284-\>317, 286-\>317, 297-301, 378-406, 593, 616, 692-693, 702-709, 761-762, 771-778 |
| src/haystack\_integrations/components/rankers/cohere/ranker.py                      |       72 |        2 |       14 |        1 |     97% |   149-154 |
| **TOTAL**                                                                           |  **624** |   **99** |  **200** |   **29** | **80%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-cohere/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-cohere/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-cohere%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.