# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       63 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      541 |      328 |      178 |       28 |     38% |87-91, 111-\>exit, 132, 142, 153-159, 170-\>exit, 187-213, 233-244, 251-255, 276, 278, 280, 282, 301-306, 316-329, 344, 348-\>347, 354, 380, 382-\>393, 396-407, 415-417, 427-431, 443-449, 463-469, 488-\>496, 509-519, 531-536, 538, 547-575, 578, 580-\>588, 615-639, 668-693, 701-709, 717-720, 730-733, 743-779, 791-826, 841-866, 883-908, 917-950, 959-991, 1007-1017, 1035-1045, 1063-1073, 1093-1103, 1113-1122, 1134-1143, 1156-1165, 1182-1191, 1219-1223, 1253-1257, 1273-1279, 1297-1303, 1323-1329, 1351-1357, 1402-\>1405, 1448-\>1453, 1454, 1457 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |       20 |       36 |        9 |     72% |60, 68, 74-89, 101, 114, 119-122, 141 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                             |  **706** |  **348** |  **218** |   **37** | **48%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-chroma/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-chroma/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-chroma%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.