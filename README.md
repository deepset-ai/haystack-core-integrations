# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       59 |       14 |        4 |        1 |     76% |91-93, 118-120, 205-210, 234-239, 255-\>258 |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      512 |      417 |      168 |        3 |     14% |85-89, 108-\>exit, 119-159, 166-208, 219-230, 237-241, 261-270, 284-292, 302-315, 329-342, 362-390, 398-400, 410-414, 426-432, 446-452, 462-479, 492-502, 509-571, 592-600, 623-631, 639-642, 652-655, 665-701, 713-748, 763-788, 805-830, 839-871, 880-912, 928-938, 956-966, 984-994, 1014-1024, 1034-1043, 1055-1064, 1077-1086, 1103-1112, 1140-1144, 1174-1178, 1194-1200, 1218-1224, 1244-1254, 1276-1286, 1324-1351, 1360-1390 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |       70 |       36 |        0 |     11% |53-89, 96-103, 110-124, 128-141, 145-167 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        3 |        0 |        0 |     73% |     50-52 |
| **TOTAL**                                                             |  **673** |  **504** |  **208** |    **4** | **20%** |           |


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