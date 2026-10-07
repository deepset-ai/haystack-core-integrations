# Repository Coverage (cohere-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere-combined/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/cohere/document\_embedder.py        |       78 |        1 |       16 |        1 |     98% |       242 |
| src/haystack\_integrations/components/embedders/cohere/document\_image\_embedder.py |      112 |        0 |       26 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/cohere/embedding\_types.py          |       17 |        3 |        2 |        1 |     79% | 25, 35-36 |
| src/haystack\_integrations/components/embedders/cohere/text\_embedder.py            |       54 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/cohere/utils.py                     |       30 |        0 |       16 |        1 |     98% | 121-\>101 |
| src/haystack\_integrations/components/generators/cohere/chat/chat\_generator.py     |      267 |       18 |      122 |       18 |     90% |80-83, 134-\>131, 144-145, 148-155, 177-\>176, 185-\>184, 194-\>196, 246-\>249, 253-\>256, 260-\>318, 263-\>318, 266-\>318, 270-\>318, 285-\>318, 287-\>318, 298-302, 399-\>383, 596, 619 |
| src/haystack\_integrations/components/rankers/cohere/ranker.py                      |       73 |        2 |       14 |        1 |     97% |   152-157 |
| src/haystack\_integrations/utils/cohere/api\_base\_url.py                           |       11 |        0 |        2 |        0 |    100% |           |
| **TOTAL**                                                                           |  **642** |   **24** |  **204** |   **22** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-cohere-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-cohere-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-cohere-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-cohere-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.