# Repository Coverage (cohere)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/cohere/document\_embedder.py        |       79 |        1 |       16 |        1 |     98% |       244 |
| src/haystack\_integrations/components/embedders/cohere/document\_image\_embedder.py |      113 |        0 |       26 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/cohere/embedding\_types.py          |       17 |        3 |        2 |        1 |     79% | 25, 35-36 |
| src/haystack\_integrations/components/embedders/cohere/text\_embedder.py            |       55 |        9 |        6 |        1 |     84% |109-\>exit, 171-183, 204-217 |
| src/haystack\_integrations/components/embedders/cohere/utils.py                     |       30 |       13 |       16 |        2 |     50% |60-\>58, 64, 97-124 |
| src/haystack\_integrations/components/generators/cohere/chat/chat\_generator.py     |      267 |       61 |      122 |       25 |     72% |60, 80-83, 95, 103-110, 134-\>131, 142-155, 175-\>181, 177-\>176, 183-190, 194-\>196, 204, 246-250, 253-\>256, 260-\>318, 263-\>318, 266-\>318, 270-\>318, 285-\>318, 287-\>318, 298-302, 379-407, 596, 619, 695-696, 705-712, 764-765, 774-781 |
| src/haystack\_integrations/components/rankers/cohere/ranker.py                      |       74 |        2 |       14 |        1 |     97% |   153-158 |
| src/haystack\_integrations/utils/cohere/api\_base\_url.py                           |       11 |        0 |        2 |        0 |    100% |           |
| **TOTAL**                                                                           |  **646** |   **89** |  **204** |   **31** | **82%** |           |


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