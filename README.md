# Repository Coverage (elasticsearch-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

| Name                                                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/elasticsearch/bm25\_retriever.py      |       36 |        2 |        4 |        1 |     92% |     73-74 |
| src/haystack\_integrations/components/retrievers/elasticsearch/embedding\_retriever.py |       35 |        2 |        4 |        1 |     92% |     72-73 |
| src/haystack\_integrations/components/retrievers/elasticsearch/sql\_retriever.py       |       53 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/elasticsearch/document\_store.py           |      513 |       43 |      154 |       27 |     89% |133-134, 293, 295, 365->357, 380-381, 398-399, 419, 446->451, 460->469, 463, 535->540, 547->557, 578->577, 581-582, 637-639, 729-731, 757-759, 785-787, 820-822, 855-857, 880-881, 902, 907-910, 937-938, 966->965, 990-991, 993->996, 1096->1095, 1133->1132, 1135->1132, 1169, 1211 |
| src/haystack\_integrations/document\_stores/elasticsearch/filters.py                   |      135 |        5 |       72 |        4 |     96% |15-16, 51, 70, 73 |
| **TOTAL**                                                                              |  **772** |   **52** |  **248** |   **33** | **91%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-elasticsearch-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-elasticsearch-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-elasticsearch-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.