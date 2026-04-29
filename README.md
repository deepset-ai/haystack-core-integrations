# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       59 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      542 |      344 |      186 |       26 |     35% |85-89, 108-\>exit, 127-159, 166-\>exit, 183-208, 219-230, 237-241, 262, 264, 266, 268, 287-292, 302-315, 330, 334-\>333, 340, 365, 367-\>376, 379-390, 398-400, 410-414, 426-432, 446-452, 471-\>479, 492-502, 514-519, 521, 530-558, 561, 563-\>571, 598-622, 651-676, 684-692, 700-703, 713-716, 726-762, 774-809, 824-849, 866-891, 900-932, 941-973, 989-999, 1017-1027, 1045-1055, 1075-1085, 1095-1104, 1116-1125, 1138-1147, 1164-1173, 1201-1205, 1235-1239, 1255-1261, 1279-1285, 1305-1315, 1337-1347, 1391-\>1394, 1437-\>1442, 1443, 1446 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |       20 |       36 |        9 |     72% |60, 68, 74-89, 101, 114, 119-122, 141 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                             |  **703** |  **364** |  **226** |   **35** | **45%** |           |


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