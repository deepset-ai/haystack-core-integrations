# Repository Coverage (google_genai-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-google_genai-combined/htmlcov/index.html)

| Name                                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/common/google\_genai/utils.py                             |       32 |        0 |       16 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/google\_genai/document\_embedder.py             |       95 |        4 |       30 |        5 |     93% |211-\>214, 222, 242-\>245, 253, 305-309 |
| src/haystack\_integrations/components/embedders/google\_genai/multimodal\_document\_embedder.py |      144 |        6 |       52 |        7 |     93% |310-311, 330-\>333, 352-\>326, 372-\>375, 383-384, 387-392, 394-\>368 |
| src/haystack\_integrations/components/embedders/google\_genai/text\_embedder.py                 |       46 |        0 |        4 |        1 |     98% | 170-\>173 |
| src/haystack\_integrations/components/generators/google\_genai/chat/chat\_generator.py          |      155 |        1 |       36 |        1 |     99% |       560 |
| src/haystack\_integrations/components/generators/google\_genai/chat/utils.py                    |      289 |       13 |      166 |       15 |     93% |222-223, 261-263, 270-\>278, 283-\>288, 327-329, 390-\>288, 453, 463-465, 503-\>537, 506-\>537, 533-\>507, 606-\>610, 653-\>624, 717-\>723, 723-\>709, 731 |
| **TOTAL**                                                                                       |  **761** |   **24** |  **304** |   **29** | **95%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-google_genai-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-google_genai-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-google_genai-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-google_genai-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-google_genai-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-google_genai-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.