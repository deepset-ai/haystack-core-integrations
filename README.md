# Repository Coverage (alloydb)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

| Name                                                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/alloydb/embedding\_retriever.py |       33 |        4 |        4 |        1 |     86% |83-90, 118 |
| src/haystack\_integrations/components/retrievers/alloydb/keyword\_retriever.py   |       32 |        8 |        4 |        0 |     72% |70-76, 100-104 |
| src/haystack\_integrations/document\_stores/alloydb/converters.py                |       42 |        3 |       18 |        4 |     88% |35-\>45, 38, 68, 71 |
| src/haystack\_integrations/document\_stores/alloydb/document\_store.py           |      479 |      328 |      128 |        9 |     28% |269-271, 287-\>exit, 292-\>exit, 332, 341-403, 409-429, 435-473, 482-490, 500-538, 546-575, 583-598, 618-640, 646-657, 676-707, 715-725, 736-743, 757-789, 800-839, 842-850, 860-873, 906-924, 937-939, 959-971, 981-989, 999-1014, 1017, 1038-1048, 1056-1072, 1090-1106, 1119-1120, 1122-1126, 1134-1166, 1185-1198, 1208-1219, 1242-1261, 1282-1313, 1351-1374 |
| src/haystack\_integrations/document\_stores/alloydb/filters.py                   |      152 |       55 |       64 |        8 |     62% |34-36, 86-\>89, 92, 115, 158, 165, 169-181, 185-197, 201-213, 217-229, 234-235, 240-245, 252, 259 |
| **TOTAL**                                                                        |  **738** |  **398** |  **218** |   **22** | **43%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-alloydb/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-alloydb/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-alloydb%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.