# Repository Coverage (alloydb)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

| Name                                                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/alloydb/embedding\_retriever.py |       31 |        4 |        4 |        1 |     86% |83-90, 118 |
| src/haystack\_integrations/components/retrievers/alloydb/keyword\_retriever.py   |       30 |        8 |        4 |        0 |     71% |70-76, 100-104 |
| src/haystack\_integrations/document\_stores/alloydb/converters.py                |       42 |        3 |       18 |        4 |     88% |35-\>45, 38, 68, 71 |
| src/haystack\_integrations/document\_stores/alloydb/document\_store.py           |      466 |      319 |      126 |        9 |     28% |268-269, 275-279, 301-\>exit, 306-\>exit, 346, 355-417, 423-443, 449-487, 496-504, 514-552, 560-589, 597-612, 632-654, 660-671, 690-721, 729-739, 750-757, 771-803, 814-853, 856-864, 874-887, 920-938, 951-953, 973-985, 995-1003, 1013-1028, 1031, 1052-1062, 1070-1086, 1104-1120, 1133-1134, 1136-1140, 1148-1180, 1199-1212, 1222-1233, 1256-1275, 1285-1312 |
| src/haystack\_integrations/document\_stores/alloydb/filters.py                   |      149 |       79 |       62 |        5 |     43% |34-36, 48, 70-90, 111, 154, 161, 165-177, 181-193, 197-209, 213-225, 229-232, 236-241, 245-248, 252-255 |
| **TOTAL**                                                                        |  **718** |  **413** |  **214** |   **19** | **39%** |           |


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