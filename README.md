# Repository Coverage (sentence_transformers-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers-combined/htmlcov/index.html)

| Name                                                                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/backend.py                              |       43 |        2 |       10 |        1 |     94% |   144-145 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/multivector\_backend.py                 |       23 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/sparse\_backend.py                      |       29 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_doc\_image\_embedder.py            |       77 |        0 |       22 |        1 |     99% | 184-\>186 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_document\_embedder.py              |       69 |        1 |       18 |        1 |     98% |       270 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_multivector\_document\_embedder.py |       65 |        1 |       18 |        0 |     99% |       151 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_multivector\_text\_embedder.py     |       48 |        1 |       10 |        0 |     98% |       126 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_document\_embedder.py      |       61 |        1 |       16 |        0 |     99% |       138 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_text\_embedder.py          |       50 |        1 |       12 |        0 |     98% |       115 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_text\_embedder.py                  |       59 |        0 |       14 |        0 |    100% |           |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_diversity.py                         |      152 |        3 |       40 |        3 |     97% |210-\>exit, 400-401, 436 |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_similarity.py                        |       79 |        0 |       20 |        0 |    100% |           |
| **TOTAL**                                                                                                                         |  **755** |   **10** |  **186** |    **6** | **98%** |           |


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