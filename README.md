# Repository Coverage (anthropic)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

| Name                                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/generators/anthropic/chat/chat\_generator.py          |      215 |       13 |       92 |       18 |     89% |240, 306, 348-349, 403-\>402, 407-\>410, 421, 430-\>436, 432, 438-\>440, 441, 483-\>482, 492-\>495, 497, 507-509, 518-\>524, 520, 524-\>531, 526-\>528, 529 |
| src/haystack\_integrations/components/generators/anthropic/chat/foundry\_chat\_generator.py |       69 |        3 |       20 |        4 |     92% |189, 194, 255-\>257, 258 |
| src/haystack\_integrations/components/generators/anthropic/chat/utils.py                    |      292 |       34 |      184 |       23 |     85% |86-\>76, 93, 95, 98-\>76, 186-\>197, 195-196, 267-271, 284-299, 303-304, 308, 316, 336-\>335, 376-\>381, 381-\>384, 407-408, 414-415, 422-\>425, 489, 554-562, 567-\>569, 570, 572, 589-590 |
| src/haystack\_integrations/components/generators/anthropic/chat/vertex\_chat\_generator.py  |       52 |        0 |       10 |        1 |     98% | 209-\>212 |
| src/haystack\_integrations/components/generators/anthropic/generator.py                     |       97 |       35 |       42 |        4 |     52% |117, 150-\>152, 176, 199-236, 245-\>267 |
| src/haystack\_integrations/token\_counters/anthropic/token\_counter.py                      |       50 |        4 |       16 |        3 |     89% |69, 71, 93-94 |
| **TOTAL**                                                                                   |  **775** |   **89** |  **364** |   **53** | **83%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-anthropic/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-anthropic/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-anthropic%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.