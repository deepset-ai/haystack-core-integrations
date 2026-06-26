# Repository Coverage (huggingface_api-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

| Name                                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/huggingface\_api/utils.py                               |       52 |        2 |       10 |        1 |     95% |   108-109 |
| src/haystack\_integrations/components/embedders/huggingface\_api/document\_embedder.py    |      126 |        6 |       34 |        5 |     93% |179-180, 259-\>263, 263-\>267, 346-350, 379-383 |
| src/haystack\_integrations/components/embedders/huggingface\_api/text\_embedder.py        |       84 |        4 |       28 |        6 |     91% |143-144, 167-\>177, 168-\>172, 172-\>177, 267, 269 |
| src/haystack\_integrations/components/generators/huggingface\_api/chat/chat\_generator.py |      232 |        8 |       86 |       14 |     93% |135-\>139, 137-\>139, 187, 220-\>222, 422-423, 490, 581-\>584, 593-594, 631-\>634, 647, 669-\>674, 702-\>705, 718, 737-\>742 |
| src/haystack\_integrations/components/rankers/huggingface\_api/ranker.py                  |       91 |        9 |       24 |        4 |     89% |219-\>223, 233-235, 275-276, 283, 286-\>290, 300-302 |
| **TOTAL**                                                                                 |  **585** |   **29** |  **182** |   **30** | **92%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-huggingface_api-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-huggingface_api-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-huggingface_api-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.