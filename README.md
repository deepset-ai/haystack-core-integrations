# Repository Coverage (astra)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/astra/retriever.py  |       39 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/astra\_client.py   |       37 |        0 |       10 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/document\_store.py |      476 |       15 |      166 |       19 |     95% |197-\>exit, 215-\>221, 245, 262-\>265, 265-\>270, 433, 444, 446, 447-\>442, 453-\>440, 457, 525, 529-532, 595, 695, 885, 888-\>891, 924, 953 |
| src/haystack\_integrations/document\_stores/astra/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/filters.py         |      114 |       19 |       54 |       12 |     82% |68, 70, 74, 86-87, 108-109, 128, 130-134, 139, 146, 153, 160, 174-175 |
| **TOTAL**                                                            |  **674** |   **34** |  **234** |   **31** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-astra/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-astra/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-astra%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.