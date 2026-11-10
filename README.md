# Repository Coverage (sentence_transformers)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers/htmlcov/index.html)

| Name                                                                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/backend.py                              |       43 |        2 |       10 |        1 |     94% |   144-145 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/multivector\_backend.py                 |       23 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/embedding\_backend/sparse\_backend.py                      |       29 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_doc\_image\_embedder.py            |       77 |        1 |       22 |        2 |     97% |184-\>186, 241 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_document\_embedder.py              |       69 |        2 |       18 |        1 |     97% |  178, 270 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_multivector\_document\_embedder.py |       65 |        2 |       18 |        1 |     96% |  151, 240 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_multivector\_text\_embedder.py     |       48 |        2 |       10 |        1 |     95% |  126, 210 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_document\_embedder.py      |       61 |        2 |       16 |        1 |     96% |  138, 224 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_sparse\_text\_embedder.py          |       50 |        2 |       12 |        1 |     95% |  115, 198 |
| src/haystack\_integrations/components/embedders/sentence\_transformers/sentence\_transformers\_text\_embedder.py                  |       59 |        2 |       14 |        1 |     96% |  163, 253 |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_diversity.py                         |      152 |        7 |       40 |        4 |     93% |210-\>exit, 400-401, 422, 435-438 |
| src/haystack\_integrations/components/rankers/sentence\_transformers/sentence\_transformers\_similarity.py                        |       79 |        2 |       20 |        2 |     96% |152, 158-\>exit, 260 |
| **TOTAL**                                                                                                                         |  **755** |   **24** |  **186** |   **15** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-sentence_transformers/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-sentence_transformers/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-sentence_transformers%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-sentence_transformers/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.