# Repository Coverage (sentence_transformers-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers-combined/htmlcov/index.html)

| Name                                                                                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/backend.py                         |       28 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/sparse\_backend.py                 |       28 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_doc\_image\_embedder.py       |       77 |        0 |       22 |        1 |     99% | 179-\>181 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_document\_embedder.py         |       69 |        1 |       18 |        1 |     98% |       265 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_document\_embedder.py |       61 |        1 |       16 |        0 |     99% |       133 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_text\_embedder.py     |       50 |        1 |       12 |        0 |     98% |       110 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_text\_embedder.py             |       59 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_diversity.py                    |      151 |        5 |       40 |        5 |     95% |202-\>exit, 242, 257, 388-389, 424 |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_similarity.py                   |       78 |        0 |       20 |        0 |    100% |           |
| **TOTAL**                                                                                                                    |  **601** |    **8** |  **150** |    **7** | **98%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-sentence_transformers-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-sentence_transformers-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-sentence_transformers-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.