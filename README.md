# Repository Coverage (alloydb)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-alloydb/htmlcov/index.html)

| Name                                                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/alloydb/embedding\_retriever.py |       33 |        4 |        4 |        1 |     86% |83-90, 118 |
| src/haystack\_integrations/components/retrievers/alloydb/keyword\_retriever.py   |       32 |        8 |        4 |        0 |     72% |70-76, 100-104 |
| src/haystack\_integrations/document\_stores/alloydb/converters.py                |       42 |        3 |       18 |        4 |     88% |35-\>45, 38, 68, 71 |
| src/haystack\_integrations/document\_stores/alloydb/document\_store.py           |      461 |      315 |      126 |        9 |     28% |270-272, 288-\>exit, 293-\>exit, 333, 342-404, 410-430, 436-474, 483-491, 501-539, 547-576, 584-599, 619-641, 647-658, 677-708, 716-726, 737-744, 758-790, 801-840, 843-851, 861-874, 907-925, 938-940, 960-972, 982-990, 1000-1015, 1018, 1039-1049, 1057-1073, 1091-1107, 1120-1121, 1123-1127, 1135-1167, 1186-1199, 1209-1220, 1243-1262, 1272-1299 |
| src/haystack\_integrations/document\_stores/alloydb/filters.py                   |      149 |       79 |       62 |        5 |     43% |34-36, 48, 70-90, 111, 154, 161, 165-177, 181-193, 197-209, 213-225, 229-232, 236-241, 245-248, 252-255 |
| **TOTAL**                                                                        |  **717** |  **409** |  **214** |   **19** | **39%** |           |


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