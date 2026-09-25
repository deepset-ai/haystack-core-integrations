# Repository Coverage (agent_pack)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-agent_pack/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/agent\_pack/advanced\_rag/agent.py                      |       36 |        0 |       10 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/advanced\_rag/hooks.py                      |       41 |        0 |        8 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/advanced\_rag/prompts.py                    |       10 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/advanced\_rag/tools.py                      |      221 |        0 |       66 |        1 |     99% | 663-\>678 |
| src/haystack\_integrations/agent\_pack/deep\_research/agent.py                     |       43 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/deep\_research/hooks.py                     |       42 |        0 |        2 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/deep\_research/prompts.py                   |        5 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/deep\_research/tools.py                     |       51 |        0 |        6 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/evaluation/dataclasses.py                   |       47 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/agent\_pack/evaluation/harness\_log\_collector.py       |       52 |        3 |        8 |        1 |     93% |74-75, 107 |
| src/haystack\_integrations/agent\_pack/evaluation/retrieval\_harness\_evaluator.py |       93 |        2 |       24 |        2 |     97% |  153, 314 |
| src/haystack\_integrations/agent\_pack/evaluation/tracer.py                        |      113 |        0 |       26 |        0 |    100% |           |
| **TOTAL**                                                                          |  **754** |    **5** |  **158** |    **4** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-agent_pack/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-agent_pack/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-agent_pack/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-agent_pack/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-agent_pack%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-agent_pack/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.