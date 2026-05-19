# Repository Coverage (vespa-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa-combined/htmlcov/index.html)

| Name                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/vespa/embedding\_retriever.py |       20 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/vespa/keyword\_retriever.py   |       18 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/vespa/document\_store.py           |      237 |       16 |       80 |       15 |     90% |38, 130-131, 243, 309, 364-365, 399-400, 437-\>439, 450, 481-\>483, 492, 510, 517-518, 525-\>528, 544, 547 |
| src/haystack\_integrations/document\_stores/vespa/errors.py                    |        5 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/vespa/filters.py                   |      109 |       22 |       64 |       12 |     78% |20, 32, 35, 48, 66-69, 82, 92, 114-120, 132, 138-139, 148-149, 161-162 |
| **TOTAL**                                                                      |  **389** |   **38** |  **148** |   **27** | **87%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vespa-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vespa-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-vespa-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.