# Repository Coverage (github)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-github/htmlcov/index.html)

| Name                                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/connectors/github/file\_editor.py     |      132 |        4 |       28 |        6 |     94% |117-\>121, 207, 224, 241, 274-\>277, 285 |
| src/haystack\_integrations/components/connectors/github/issue\_commenter.py |       61 |        1 |       12 |        2 |     96% |69-\>73, 116 |
| src/haystack\_integrations/components/connectors/github/issue\_viewer.py    |       66 |        0 |       10 |        1 |     99% | 198-\>202 |
| src/haystack\_integrations/components/connectors/github/pr\_creator.py      |      138 |       41 |       20 |        1 |     71% |72-\>76, 156-196, 208-218 |
| src/haystack\_integrations/components/connectors/github/repo\_forker.py     |      122 |        2 |       24 |        3 |     97% |87-\>91, 156, 292 |
| src/haystack\_integrations/components/connectors/github/repo\_viewer.py     |       88 |        1 |       24 |        3 |     96% |114, 159-\>162, 221-\>224 |
| src/haystack\_integrations/prompts/github/context\_prompt.py                |        1 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/file\_editor\_prompt.py           |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/issue\_commenter\_prompt.py       |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/issue\_viewer\_prompt.py          |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/pr\_creator\_prompt.py            |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/repo\_forker\_prompt.py           |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/repo\_viewer\_prompt.py           |        2 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/prompts/github/system\_prompt.py                 |        1 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/file\_editor\_tool.py               |       29 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/issue\_commenter\_tool.py           |       28 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/issue\_viewer\_tool.py              |       28 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/pr\_creator\_tool.py                |       27 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/repo\_forker\_tool.py               |       27 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/repo\_viewer\_tool.py               |       30 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tools/github/utils.py                            |       40 |        0 |       26 |        0 |    100% |           |
| **TOTAL**                                                                   |  **830** |   **49** |  **144** |   **16** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-github/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-github/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-github/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-github/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-github%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-github/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.