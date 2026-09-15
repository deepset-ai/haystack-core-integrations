# Repository Coverage (google_genai-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-google_genai-combined/htmlcov/index.html)

| Name                                                                                            |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/common/google\_genai/utils.py                                        |       38 |        0 |       20 |        2 |     97% |16-\>exit, 29-\>exit |
| src/haystack\_integrations/components/embedders/google\_genai/document\_embedder.py             |      122 |        5 |       38 |        4 |     94% |274, 297, 307, 365-369 |
| src/haystack\_integrations/components/embedders/google\_genai/multimodal\_document\_embedder.py |      180 |        8 |       64 |        7 |     94% |405-406, 428, 449-\>422, 472, 482-483, 486-491, 493-\>466 |
| src/haystack\_integrations/components/embedders/google\_genai/text\_embedder.py                 |       68 |        0 |       12 |        0 |    100% |           |
| src/haystack\_integrations/components/generators/google\_genai/chat/chat\_generator.py          |      185 |        1 |       46 |        2 |     99% |421-\>423, 626 |
| src/haystack\_integrations/components/generators/google\_genai/chat/utils.py                    |      308 |       14 |      180 |       15 |     93% |222-223, 264-266, 273-\>281, 286-\>291, 330-332, 401-\>291, 464, 474-476, 514-\>548, 517-\>548, 544-\>518, 676-\>647, 749-\>755, 755-\>741, 765, 771 |
| src/haystack\_integrations/token\_counters/google\_genai/token\_counter.py                      |       59 |        2 |       18 |        2 |     95% |146-147, 156-\>exit |
| **TOTAL**                                                                                       |  **960** |   **30** |  **378** |   **32** | **95%** |           |


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