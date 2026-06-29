# Repository Coverage (chroma)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-chroma/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/chroma/retriever.py  |       59 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/document\_store.py |      540 |      333 |      182 |       28 |     37% |85-89, 108-\>exit, 129, 139, 150-156, 167-\>exit, 184-210, 221-232, 239-243, 264, 266, 268, 270, 289-294, 304-317, 332, 336-\>335, 342, 367, 369-\>378, 381-392, 400-402, 412-416, 428-434, 448-454, 473-\>481, 494-504, 516-521, 523, 532-560, 563, 565-\>573, 600-624, 653-678, 686-694, 702-705, 715-718, 728-764, 776-811, 826-851, 868-893, 902-934, 943-975, 991-1001, 1019-1029, 1047-1057, 1077-1087, 1097-1106, 1118-1127, 1140-1149, 1166-1175, 1203-1207, 1237-1241, 1257-1263, 1281-1287, 1307-1317, 1339-1349, 1394-\>1397, 1440-\>1445, 1446, 1449 |
| src/haystack\_integrations/document\_stores/chroma/errors.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/chroma/filters.py         |       83 |       20 |       36 |        9 |     72% |60, 68, 74-89, 101, 114, 119-122, 141 |
| src/haystack\_integrations/document\_stores/chroma/utils.py           |       11 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                             |  **701** |  **353** |  **222** |   **37** | **47%** |           |


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