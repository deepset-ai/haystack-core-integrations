# Repository Coverage (transformers)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-transformers/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/transformers/utils.py                                            |       92 |       10 |       24 |        6 |     84% |96, 99-100, 127-131, 136, 187-190, 224-\>exit, 234-\>236 |
| src/haystack\_integrations/components/classifiers/transformers/zero\_shot\_document\_classifier.py |       58 |        3 |       18 |        1 |     92% |138-140, 146-\>exit |
| src/haystack\_integrations/components/extractors/transformers/named\_entity\_extractor.py          |       82 |       23 |       10 |        1 |     70% |117-137, 154, 185-191, 249, 262 |
| src/haystack\_integrations/components/generators/transformers/chat/chat\_generator.py              |      213 |       19 |       60 |       10 |     89% |64-66, 209-\>214, 265-\>exit, 272-274, 284-\>288, 333, 336-\>339, 364, 422, 451-456, 483, 536-539, 575-581 |
| src/haystack\_integrations/components/readers/transformers/extractive\_reader.py                   |      229 |        9 |       68 |        9 |     94% |131, 180-\>exit, 193, 218-223, 343-\>348, 415, 472, 494-\>461, 585, 630-631 |
| src/haystack\_integrations/components/routers/transformers/text\_router.py                         |       51 |        6 |       16 |        4 |     82% |107, 116-118, 124-\>128, 129-\>exit, 132-136 |
| src/haystack\_integrations/components/routers/transformers/zero\_shot\_text\_router.py             |       46 |        3 |       10 |        1 |     89% |140-142, 148-\>exit |
| **TOTAL**                                                                                          |  **771** |   **73** |  **206** |   **32** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-transformers/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-transformers/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-transformers/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-transformers/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-transformers%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-transformers/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.