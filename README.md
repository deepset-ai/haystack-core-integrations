# Repository Coverage (rhesis)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-rhesis/htmlcov/index.html)

| Name                                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/connectors/rhesis/rhesis\_connector.py |       60 |        6 |       10 |        1 |     87% |27-31, 213 |
| src/haystack\_integrations/tracing/rhesis/\_extraction.py                    |       62 |       14 |       38 |        5 |     73% |25-33, 51-53, 61, 95, 119-\>124 |
| src/haystack\_integrations/tracing/rhesis/\_haystack\_tags.py                |       21 |        0 |        0 |        0 |    100% |           |
| src/haystack\_integrations/tracing/rhesis/conversation.py                    |       89 |        6 |       16 |        3 |     91% |50-52, 157-158, 193, 231-\>234 |
| src/haystack\_integrations/tracing/rhesis/mapping.py                         |       45 |        1 |       20 |        1 |     97% |        75 |
| src/haystack\_integrations/tracing/rhesis/tracer.py                          |      497 |       59 |      198 |       39 |     84% |122-\>127, 127-\>exit, 153-154, 175, 234-\>239, 245-\>247, 253, 265-\>271, 275-\>277, 284, 293-304, 316, 338-\>exit, 353-354, 376-377, 379-380, 382-383, 399, 404, 409, 419-424, 433-436, 456, 488-492, 494-\>496, 512, 516, 528-\>532, 533-\>535, 536-\>538, 547-552, 588-589, 594-\>596, 613-\>exit, 687-\>686, 692, 697-\>700, 707-\>706, 733-\>exit, 738-\>exit, 742-747, 751-\>exit, 754-\>exit, 840-841, 853-854, 864-865, 886 |
| **TOTAL**                                                                    |  **774** |   **86** |  **282** |   **49** | **85%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-rhesis/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-rhesis/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-rhesis/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-rhesis/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-rhesis%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-rhesis/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.