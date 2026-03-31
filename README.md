# Repository Coverage (opensearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/opensearch/bm25\_retriever.py                 |       79 |       12 |       28 |        6 |     81% |102-103, 171, 180->183, 273-274, 283, 340-341, 348-351 |
| src/haystack\_integrations/components/retrievers/opensearch/embedding\_retriever.py            |       85 |       15 |       36 |       12 |     76% |110-111, 248, 250, 251->253, 262-263, 279, 374, 376, 377->379, 379->381, 381->384, 388-389, 403-406 |
| src/haystack\_integrations/components/retrievers/opensearch/metadata\_retriever.py             |       82 |        4 |       16 |        2 |     94% |257-258, 371-372 |
| src/haystack\_integrations/components/retrievers/opensearch/open\_search\_hybrid\_retriever.py |       77 |        0 |       14 |        3 |     97% |346->350, 350->354, 354->358 |
| src/haystack\_integrations/components/retrievers/opensearch/sql\_retriever.py                  |       53 |        4 |       14 |        2 |     91% |113-114, 168-169 |
| src/haystack\_integrations/document\_stores/opensearch/auth.py                                 |       63 |        7 |        4 |        0 |     90% |66-69, 168-173 |
| src/haystack\_integrations/document\_stores/opensearch/document\_store.py                      |      664 |       51 |      206 |       26 |     90% |193->195, 195->197, 197->200, 200->exit, 254->257, 272->274, 276, 363, 440->450, 443, 565-571, 702-704, 739-741, 770-772, 799-801, 839-841, 879-881, 894-896, 943-946, 1141-1143, 1171, 1173->1172, 1281, 1300-1302, 1305->1309, 1386, 1405-1407, 1410->1414, 1427-1428, 1601->1600, 1639->1638, 1641->1638, 1669-1670, 1675, 1718 |
| src/haystack\_integrations/document\_stores/opensearch/filters.py                              |      135 |        5 |       72 |        4 |     96% |15-16, 51, 70, 73 |
| src/haystack\_integrations/document\_stores/opensearch/opensearch\_scripts.py                  |        1 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                      | **1239** |   **98** |  **390** |   **55** | **90%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-opensearch-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-opensearch-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-opensearch-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.