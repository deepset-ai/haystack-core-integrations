# Repository Coverage (pgvector-combined)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

| Name                                                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/pgvector/embedding\_retriever.py |       44 |        4 |        6 |        3 |     86% |89-90, 93-94, 134->136 |
| src/haystack\_integrations/components/retrievers/pgvector/keyword\_retriever.py   |       37 |        2 |        4 |        1 |     93% |     69-70 |
| src/haystack\_integrations/document\_stores/pgvector/converters.py                |       42 |        2 |       18 |        3 |     92% |31->41, 34, 67 |
| src/haystack\_integrations/document\_stores/pgvector/document\_store.py           |      716 |       81 |      182 |       44 |     86% |163-164, 167-168, 237-238, 248-249, 253->exit, 258->exit, 281-285, 292-297, 302->exit, 307->exit, 330-334, 341-346, 364-365, 378->380, 421->423, 612-613, 617-618, 635-636, 654, 668-673, 691, 707-712, 740, 765, 871->876, 895-901, 928->933, 930-931, 952-958, 973, 996, 1062->1066, 1084-1086, 1104->1108, 1126-1128, 1153->1158, 1176-1178, 1203->1208, 1226-1228, 1343-1344, 1356->1359, 1454->1457, 1480, 1505, 1521-1525, 1573, 1593-1594, 1627-1628, 1655, 1662, 1678, 1688, 1832-1833, 1837-1838, 1889-1890 |
| src/haystack\_integrations/document\_stores/pgvector/filters.py                   |      149 |        4 |       62 |        3 |     97% |82->85, 246-247, 253-254 |
| **TOTAL**                                                                         |  **988** |   **93** |  **272** |   **54** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector-combined/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-pgvector-combined/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-pgvector-combined%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-pgvector-combined/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.