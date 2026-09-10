# Repository Coverage (transformers)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-transformers/htmlcov/index.html)

| Name                                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/transformers/utils.py                                            |       88 |        7 |       22 |        4 |     88% |126-130, 135, 186-189, 223-\>exit, 233-\>235 |
| src/haystack\_integrations/components/classifiers/transformers/zero\_shot\_document\_classifier.py |       60 |        3 |       16 |        0 |     93% |   136-138 |
| src/haystack\_integrations/components/extractors/transformers/named\_entity\_extractor.py          |       73 |        7 |        8 |        0 |     89% |169-174, 232, 245 |
| src/haystack\_integrations/components/generators/transformers/chat/chat\_generator.py              |      208 |       18 |       52 |        6 |     90% |63-65, 235-237, 312, 315-\>318, 401, 430-435, 518-521, 554, 560-566 |
| src/haystack\_integrations/components/readers/transformers/extractive\_reader.py                   |      235 |        8 |       66 |        7 |     95% |131, 191, 220-225, 347-\>352, 419, 476, 498-\>465, 636-637 |
| src/haystack\_integrations/components/routers/transformers/text\_router.py                         |       55 |        5 |       14 |        2 |     87% |113-115, 129-\>137, 132-136 |
| src/haystack\_integrations/components/routers/transformers/zero\_shot\_text\_router.py             |       48 |        3 |        8 |        0 |     91% |   138-140 |
| **TOTAL**                                                                                          |  **767** |   **51** |  **186** |   **19** | **91%** |           |


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