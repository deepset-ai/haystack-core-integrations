# Repository Coverage (vespa)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa/htmlcov/index.html)

| Name                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/vespa/embedding\_retriever.py |       20 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/vespa/keyword\_retriever.py   |       18 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/vespa/document\_store.py           |      237 |       38 |       80 |       16 |     80% |36-39, 47-48, 130-131, 186-188, 243, 276-289, 309, 336-342, 364-365, 399-400, 437-\>439, 450, 481-\>483, 492, 506, 510, 517-518, 525-\>528, 544, 547 |
| src/haystack\_integrations/document\_stores/vespa/errors.py                    |        5 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/vespa/filters.py                   |      109 |       56 |       64 |       15 |     44% |18, 20, 32, 35, 48, 51, 54-58, 66-69, 82, 90-120, 132, 136-143, 148-149, 157-158, 161-162 |
| **TOTAL**                                                                      |  **389** |   **94** |  **148** |   **31** | **70%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vespa/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-vespa/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-vespa%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-vespa/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.