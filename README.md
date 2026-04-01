# Repository Coverage (cohere)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/cohere/document\_embedder.py        |       67 |        1 |       12 |        1 |     97% |       233 |
| src/haystack\_integrations/components/embedders/cohere/document\_image\_embedder.py |      101 |        0 |       22 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/cohere/embedding\_types.py          |       17 |        3 |        2 |        1 |     79% | 25, 35-36 |
| src/haystack\_integrations/components/embedders/cohere/text\_embedder.py            |       43 |        5 |        2 |        1 |     87% |101-\>exit, 164-173, 194-205 |
| src/haystack\_integrations/components/embedders/cohere/utils.py                     |       29 |       23 |       14 |        0 |     14% |36-57, 88-115 |
| src/haystack\_integrations/components/generators/cohere/chat/chat\_generator.py     |      237 |       65 |      108 |       21 |     68% |59, 79-82, 94, 102-109, 117-\>129, 133-\>130, 141-154, 174-\>180, 176-\>175, 182-189, 193-\>195, 203, 245-249, 252-\>255, 259-\>306, 262-\>306, 265-\>306, 269-\>306, 286-290, 360-380, 577, 648-649, 658-665, 706-744 |
| src/haystack\_integrations/components/generators/cohere/generator.py                |       28 |        8 |        2 |        0 |     73% |102-107, 125-130 |
| src/haystack\_integrations/components/rankers/cohere/ranker.py                      |       50 |        2 |        8 |        1 |     95% |   148-153 |
| **TOTAL**                                                                           |  **572** |  **107** |  **170** |   **25** | **77%** |           |


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