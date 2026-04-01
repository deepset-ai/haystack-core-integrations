# Repository Coverage (amazon_bedrock-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock-combined/htmlcov/index.html)

| Name                                                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/amazon\_bedrock/errors.py                                  |        4 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/common/amazon\_bedrock/utils.py                                   |       18 |        6 |        2 |        1 |     65% |49, 65-67, 77-78 |
| src/haystack\_integrations/common/s3/errors.py                                               |        3 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/common/s3/utils.py                                                |       49 |       35 |        8 |        0 |     25% |43-53, 68-100, 124-133 |
| src/haystack\_integrations/components/downloaders/s3/s3\_downloader.py                       |       93 |       18 |       20 |        8 |     73% |94-98, 133-135, 156, 161, 176, 196-199, 205, 230-235, 272 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/document\_embedder.py        |      101 |        6 |       20 |        2 |     93% |171, 217-219, 253-254 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/document\_image\_embedder.py |      123 |       22 |       32 |        6 |     79% |235-243, 253, 261-262, 266-267, 271-291, 303, 313-315 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/text\_embedder.py            |       63 |        3 |       14 |        3 |     92% |146, 148-\>153, 173-174 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/adapters.py                 |      144 |        2 |       22 |        2 |     98% |60-\>58, 255-\>257, 407-408 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/chat/chat\_generator.py     |      153 |       48 |       34 |        3 |     67% |276-281, 298-319, 358-\>361, 362-\>364, 457-458, 508-539, 569-608 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/chat/utils.py               |      303 |       20 |      180 |       19 |     91% |87, 202, 210-\>207, 213-214, 243-\>241, 335-336, 382-383, 419-\>421, 443-\>515, 445-\>515, 485-\>471, 488-\>471, 491-\>489, 497-\>504, 500-\>504, 542-\>630, 583-\>630, 613-\>623, 627-\>630, 671-\>678, 700-713 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/generator.py                |       94 |       13 |       16 |        3 |     85% |149-151, 180-185, 220-227, 242-244, 318 |
| src/haystack\_integrations/components/rankers/amazon\_bedrock/ranker.py                      |       79 |       13 |       12 |        3 |     82% |94-95, 119-124, 196-197, 209-214, 256-257, 259-260 |
| **TOTAL**                                                                                    | **1227** |  **186** |  **360** |   **50** | **83%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-amazon_bedrock-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-amazon_bedrock-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-amazon_bedrock-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.