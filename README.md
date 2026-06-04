# Repository Coverage (supabase)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-supabase/htmlcov/index.html)

| Name                                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/downloaders/supabase/supabase\_bucket\_downloader.py |       49 |        2 |       12 |        2 |     93% |75-76, 90-\>92 |
| src/haystack\_integrations/components/retrievers/supabase/embedding\_retriever.py          |       23 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/supabase/groonga\_bm25\_retriever.py      |       60 |        0 |       14 |        1 |     99% | 181-\>183 |
| src/haystack\_integrations/components/retrievers/supabase/keyword\_retriever.py            |       23 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/supabase/document\_store.py                    |       13 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/supabase/groonga\_document\_store.py           |      393 |      252 |      192 |       19 |     32% |101-102, 105, 136-137, 140, 165-169, 202-212, 222-229, 240-248, 253-255, 270-335, 339-405, 420-421, 423-425, 428-429, 434-461, 476-477, 479-481, 486-515, 524-528, 537-541, 553-570, 580-593, 599-602, 608-609, 618-619, 623, 633-634, 651-652, 664, 682-694, 731, 743, 761 |
| **TOTAL**                                                                                  |  **561** |  **254** |  **226** |   **22** | **49%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-supabase/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-supabase/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-supabase/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-supabase/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-supabase%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-supabase/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.