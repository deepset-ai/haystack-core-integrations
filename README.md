# Repository Coverage (mariadb)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mariadb/htmlcov/index.html)

| Name                                                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/mariadb/embedding\_retriever.py |       30 |        0 |        4 |        1 |     97% |   83-\>85 |
| src/haystack\_integrations/components/retrievers/mariadb/keyword\_retriever.py   |       29 |        0 |        4 |        1 |     97% |   79-\>81 |
| src/haystack\_integrations/document\_stores/mariadb/document\_store.py           |      233 |        9 |       54 |        8 |     93% |216-217, 235-238, 242, 265, 271-\>277, 277-\>283, 293-294, 434-\>439, 458-\>460, 483-\>487, 530-\>533, 537-\>540 |
| src/haystack\_integrations/document\_stores/mariadb/filters.py                   |      135 |       20 |       60 |        9 |     85% |40-41, 43-44, 61-62, 71-72, 74-75, 114, 130, 140-141, 150-151, 164-165, 192-193 |
| **TOTAL**                                                                        |  **427** |   **29** |  **122** |   **19** | **91%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-mariadb/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mariadb/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-mariadb/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mariadb/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-mariadb%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mariadb/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.