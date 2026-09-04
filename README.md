# Repository Coverage (amazon_bedrock)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock/htmlcov/index.html)

| Name                                                                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/amazon\_bedrock/errors.py                                    |        4 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/common/amazon\_bedrock/utils.py                                     |       21 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/common/s3/errors.py                                                 |        3 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/common/s3/utils.py                                                  |       51 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/components/downloaders/s3/s3\_downloader.py                         |      117 |        3 |       22 |        1 |     97% |190, 285-286 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/document\_embedder.py          |      116 |        6 |       30 |        2 |     95% |186, 240-242, 277-278 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/document\_image\_embedder.py   |      132 |        4 |       36 |        2 |     96% |280-281, 288-292 |
| src/haystack\_integrations/components/embedders/amazon\_bedrock/text\_embedder.py              |       77 |        3 |       24 |        3 |     94% |162, 164-\>175, 195-196 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/chat/chat\_generator.py       |      198 |       11 |       54 |        2 |     94% |620-621, 702-707, 717-719 |
| src/haystack\_integrations/components/generators/amazon\_bedrock/chat/utils.py                 |      317 |        7 |      188 |       18 |     95% |89, 229-230, 259-\>257, 351-352, 398-399, 440-\>442, 464-\>536, 466-\>536, 506-\>492, 509-\>492, 512-\>510, 518-\>525, 521-\>525, 563-\>654, 607-\>654, 637-\>647, 651-\>654, 695-\>702 |
| src/haystack\_integrations/components/rankers/amazon\_bedrock/ranker.py                        |       91 |        0 |       18 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/amazon\_bedrock/knowledge\_base\_retriever.py |       91 |       21 |       28 |        4 |     67% |27-38, 166-\>208, 188-202, 204-206 |
| src/haystack\_integrations/token\_counters/amazon\_bedrock/token\_counter.py                   |       65 |        2 |       14 |        1 |     96% |   151-152 |
| **TOTAL**                                                                                      | **1283** |   **57** |  **426** |   **33** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-amazon_bedrock/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-amazon_bedrock/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-amazon_bedrock%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-amazon_bedrock/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.