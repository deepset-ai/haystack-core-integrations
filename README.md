# Repository Coverage (astra-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra-combined/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/astra/retriever.py  |       36 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/astra\_client.py   |      133 |        0 |       32 |        1 |     99% | 294-\>292 |
| src/haystack\_integrations/document\_stores/astra/document\_store.py |      305 |       34 |      146 |       22 |     86% |195-\>198, 198-\>206, 209-\>219, 264-\>262, 271-\>280, 278, 293-294, 298, 309, 311, 312-\>307, 318-\>305, 322, 352-353, 357, 360-376, 410-412, 422-427, 462-\>465, 487, 498-499, 502, 531, 585-\>588 |
| src/haystack\_integrations/document\_stores/astra/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/filters.py         |      114 |        7 |       54 |        5 |     93% |68, 70, 74, 86-87, 108-109 |
| **TOTAL**                                                            |  **596** |   **41** |  **236** |   **28** | **91%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-astra-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-astra-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-astra-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.