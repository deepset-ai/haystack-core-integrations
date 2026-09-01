# Repository Coverage (solr-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-solr-combined/htmlcov/index.html)

| Name                                                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/solr/bm25\_retriever.py         |       66 |        1 |       10 |        1 |     97% |       193 |
| src/haystack\_integrations/components/retrievers/solr/embedding\_retriever.py    |       61 |        0 |       10 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/solr/solr\_hybrid\_retriever.py |       70 |        0 |       14 |        2 |     98% |236-\>235, 238-\>241 |
| src/haystack\_integrations/document\_stores/solr/client.py                       |       91 |        0 |       24 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/solr/document\_store.py              |      548 |        0 |      150 |        1 |     99% | 359-\>368 |
| src/haystack\_integrations/document\_stores/solr/errors.py                       |        3 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/solr/filters.py                      |      130 |        0 |       68 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/solr/schema.py                       |      108 |        1 |       52 |        1 |     99% |       205 |
| **TOTAL**                                                                        | **1077** |    **2** |  **328** |    **5** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-solr-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-solr-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-solr-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-solr-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-solr-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-solr-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.