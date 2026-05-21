# Repository Coverage (mem0-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mem0-combined/htmlcov/index.html)

| Name                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/mem0/retriever.py |       20 |        0 |        2 |        1 |     95% | 105-\>109 |
| src/haystack\_integrations/components/writers/mem0/writer.py       |       21 |        0 |        2 |        1 |     96% |   89-\>91 |
| src/haystack\_integrations/memory\_stores/mem0/errors.py           |        1 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/memory\_stores/mem0/filters.py          |       65 |        2 |       30 |        1 |     97% |   148-149 |
| src/haystack\_integrations/memory\_stores/mem0/memory\_store.py    |       64 |        0 |       16 |        2 |     98% |50-\>exit, 112-\>120 |
| src/haystack\_integrations/tools/mem0/retriever\_tool.py           |       32 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/tools/mem0/writer\_tool.py              |       30 |        0 |        2 |        0 |    100% |           |
| **TOTAL**                                                          |  **233** |    **2** |   **56** |    **5** | **98%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-mem0-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mem0-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-mem0-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mem0-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-mem0-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-mem0-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.