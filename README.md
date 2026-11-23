# Repository Coverage (anthropic)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-anthropic/htmlcov/index.html)

| Name                                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/generators/anthropic/chat/chat\_generator.py          |      227 |       10 |       96 |       12 |     93% |268, 334, 385-386, 452-\>451, 469, 479-\>483, 485-\>487, 488, 530-\>529, 553-555, 565-\>569, 571-\>573, 574 |
| src/haystack\_integrations/components/generators/anthropic/chat/foundry\_chat\_generator.py |       69 |        3 |       20 |        4 |     92% |205, 210, 271-\>273, 274 |
| src/haystack\_integrations/components/generators/anthropic/chat/utils.py                    |      308 |       34 |      188 |       22 |     85% |91-\>81, 98, 100, 103-\>81, 206-\>217, 215-216, 287-291, 304-319, 323-324, 328, 336, 356-\>355, 397-\>402, 402-\>405, 464-465, 471-472, 541, 606-614, 619-\>621, 622, 624, 641-642 |
| src/haystack\_integrations/components/generators/anthropic/chat/vertex\_chat\_generator.py  |       53 |        0 |       10 |        1 |     98% | 225-\>228 |
| src/haystack\_integrations/token\_counters/anthropic/token\_counter.py                      |       49 |        4 |       16 |        3 |     89% |69, 71, 93-94 |
| **TOTAL**                                                                                   |  **706** |   **51** |  **330** |   **42** | **89%** |           |


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