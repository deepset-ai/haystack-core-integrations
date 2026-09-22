# Repository Coverage (anthropic)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

| Name                                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/generators/anthropic/chat/chat\_generator.py          |      216 |       12 |       92 |       16 |     90% |249, 315, 357-358, 412-\>411, 416-\>419, 430, 439-\>445, 441, 447-\>449, 450, 492-\>491, 501-\>504, 516-518, 527-\>533, 529, 535-\>537, 538 |
| src/haystack\_integrations/components/generators/anthropic/chat/foundry\_chat\_generator.py |       69 |        3 |       20 |        4 |     92% |198, 203, 264-\>266, 267 |
| src/haystack\_integrations/components/generators/anthropic/chat/utils.py                    |      297 |       34 |      186 |       22 |     85% |89-\>79, 96, 98, 101-\>79, 189-\>200, 198-199, 270-274, 287-302, 306-307, 311, 319, 339-\>338, 379-\>384, 384-\>387, 430-431, 437-438, 507, 572-580, 585-\>587, 588, 590, 607-608 |
| src/haystack\_integrations/components/generators/anthropic/chat/vertex\_chat\_generator.py  |       52 |        0 |       10 |        1 |     98% | 216-\>219 |
| src/haystack\_integrations/token\_counters/anthropic/token\_counter.py                      |       50 |        4 |       16 |        3 |     89% |69, 71, 93-94 |
| **TOTAL**                                                                                   |  **684** |   **53** |  **324** |   **46** | **88%** |           |


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