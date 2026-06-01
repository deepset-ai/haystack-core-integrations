# Repository Coverage (supabase)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-supabase/htmlcov/index.html)

| Name                                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/downloaders/supabase/supabase\_bucket\_downloader.py |       49 |        2 |       12 |        2 |     93% |75-76, 90-\>92 |
| src/haystack\_integrations/components/retrievers/supabase/embedding\_retriever.py          |       23 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/components/retrievers/supabase/groonga\_bm25\_retriever.py      |       38 |        0 |        6 |        1 |     98% | 151-\>153 |
| src/haystack\_integrations/components/retrievers/supabase/keyword\_retriever.py            |       23 |        0 |        4 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/supabase/document\_store.py                    |       13 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/document\_stores/supabase/groonga\_document\_store.py           |      287 |      159 |      140 |       25 |     41% |100-101, 104, 132-133, 159-160, 180, 183-184, 186, 193, 210, 216-217, 220-221, 231-274, 281-282, 285-286, 297-344, 359-360, 363-364, 367-368, 387-388, 394-398, 409-413, 425-442, 449-450, 460-461, 482-483, 495, 508-537 |
| **TOTAL**                                                                                  |  **433** |  **161** |  **166** |   **28** | **57%** |           |


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