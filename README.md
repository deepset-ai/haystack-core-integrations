# Repository Coverage (astra)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-astra/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/astra/retriever.py  |       36 |        5 |        4 |        1 |     85% |60-61, 81-84 |
| src/haystack\_integrations/document\_stores/astra/astra\_client.py   |      133 |       97 |       32 |        0 |     22% |39, 70-151, 176-187, 190-192, 201-218, 223-230, 242-257, 266-274, 284-299, 308-311, 321-337, 352-363, 371-373, 383, 393, 410-412 |
| src/haystack\_integrations/document\_stores/astra/document\_store.py |      305 |      178 |      146 |       14 |     37% |25-27, 82-86, 91-95, 175-280, 288, 293-294, 298, 309, 311, 312->307, 318->305, 322, 325-326, 351-383, 387-401, 410-412, 422-427, 438-451, 460-471, 478-487, 497-508, 522-542, 581, 585->588, 610, 628->632 |
| src/haystack\_integrations/document\_stores/astra/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/astra/filters.py         |       72 |       41 |       42 |       11 |     40% |15-16, 20, 28, 34, 37, 41-42, 46, 48, 69-83, 88-89, 93-94, 96-97, 121-132 |
| **TOTAL**                                                            |  **554** |  **321** |  **224** |   **26** | **37%** |           |


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