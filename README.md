# Repository Coverage (anthropic)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

| Name                                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/generators/anthropic/chat/chat\_generator.py          |      167 |       31 |       62 |        8 |     76% |205, 270, 303-304, 357-\>356, 361-\>364, 372, 381-\>386, 383, 419-459 |
| src/haystack\_integrations/components/generators/anthropic/chat/foundry\_chat\_generator.py |       81 |        6 |       22 |        9 |     85% |187, 192, 197, 202, 205, 233-\>235, 261-\>263, 306-\>308, 309 |
| src/haystack\_integrations/components/generators/anthropic/chat/utils.py                    |      232 |       28 |      148 |       21 |     84% |120-\>129, 127-128, 201-216, 220-221, 225, 233, 253-\>252, 284-\>290, 285-\>287, 287-\>290, 313-314, 320-321, 328-\>331, 383, 395-\>402, 447-455, 460-\>462, 463, 465, 468-\>477, 482-483 |
| src/haystack\_integrations/components/generators/anthropic/chat/vertex\_chat\_generator.py  |       43 |        0 |        6 |        1 |     98% | 188-\>191 |
| src/haystack\_integrations/components/generators/anthropic/generator.py                     |       97 |       35 |       42 |        4 |     52% |117, 150-\>152, 176, 199-236, 245-\>267 |
| **TOTAL**                                                                                   |  **620** |  **100** |  **280** |   **43** | **78%** |           |


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