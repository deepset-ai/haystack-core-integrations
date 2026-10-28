# Repository Coverage (falkordb)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-falkordb/htmlcov/index.html)

| Name                                                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/falkordb/cypher\_retriever.py           |       39 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/falkordb/embedding\_retriever.py        |       40 |        0 |        4 |        1 |     98% |   90-\>92 |
| src/haystack\_integrations/components/retrievers/falkordb/text\_to\_cypher\_retriever.py |      109 |        0 |       26 |        1 |     99% |147-\>exit |
| src/haystack\_integrations/document\_stores/falkordb/document\_store.py                  |      708 |       38 |      190 |       12 |     93% |183, 232-\>exit, 293-300, 310-318, 376-383, 392-400, 607-609, 705-706, 742, 745, 749, 809-811, 829, 1103-1104, 1533, 1554-1561, 1660-1661 |
| **TOTAL**                                                                                |  **896** |   **38** |  **226** |   **14** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-falkordb/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-falkordb/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-falkordb/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-falkordb/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-falkordb%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-falkordb/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.