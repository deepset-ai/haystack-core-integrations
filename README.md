# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       63 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      569 |      333 |      194 |       33 |     40% |87-91, 111-\>exit, 132, 142, 153-159, 170-\>exit, 187-213, 233-244, 251-255, 276, 278, 280, 282, 301-306, 316-329, 344, 348-\>347, 354, 380, 382-\>393, 398-416, 424-426, 436-440, 452-458, 472-478, 497-\>505, 518-528, 540-545, 547, 556-584, 587, 626-651, 686, 692, 699, 703, 718-729, 738, 746-749, 759-762, 772-808, 820-855, 870-895, 912-937, 946-979, 988-1020, 1036-1046, 1064-1074, 1092-1102, 1122-1132, 1142-1151, 1163-1172, 1185-1194, 1211-1220, 1248-1252, 1282-1286, 1302-1308, 1326-1332, 1354-1362, 1386-1394, 1439-\>1442, 1485-\>1490, 1491, 1494 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |        5 |       36 |        4 |     92% |59, 100, 118-121, 140 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                             |  **734** |  **338** |  **234** |   **37** | **52%** |           |


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