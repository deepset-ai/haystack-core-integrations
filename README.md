# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       63 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      556 |      343 |      186 |       28 |     37% |87-91, 111-\>exit, 132, 142, 153-159, 170-\>exit, 187-213, 233-244, 251-255, 276, 278, 280, 282, 301-306, 316-329, 344, 348-\>347, 354, 380, 382-\>393, 398-416, 424-426, 436-440, 452-458, 472-478, 497-\>505, 518-528, 540-545, 547, 556-584, 587, 589-\>597, 624-648, 677-702, 710-718, 726-729, 739-742, 752-788, 800-835, 850-875, 892-917, 926-959, 968-1000, 1016-1026, 1044-1054, 1072-1082, 1102-1112, 1122-1131, 1143-1152, 1165-1174, 1191-1200, 1228-1232, 1262-1266, 1282-1288, 1306-1312, 1334-1342, 1366-1374, 1419-\>1422, 1465-\>1470, 1471, 1474 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |       20 |       36 |        9 |     72% |60, 68, 74-89, 101, 114, 119-122, 141 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                             |  **721** |  **363** |  **226** |   **37** | **47%** |           |


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