# Repository Coverage (qdrant)

[Full report](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

| Name                                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|---------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/haystack\_integrations/components/retrievers/qdrant/retriever.py  |      125 |       28 |       24 |        6 |     72% |77-78, 129-\>131, 163-186, 218-241, 305-306, 396-419, 456-479, 548-549, 637, 697 |
| src/haystack\_integrations/document\_stores/qdrant/converters.py      |       42 |       17 |       16 |        3 |     52% |22-48, 71-\>83, 74-\>83, 76-\>83 |
| src/haystack\_integrations/document\_stores/qdrant/document\_store.py |      787 |      550 |      240 |       14 |     26% |48-52, 307-318, 324-335, 353-354, 370-371, 393-425, 447-480, 489-499, 510-520, 534-557, 569-592, 602, 620-629, 639-644, 654-663, 670-680, 691-701, 713-725, 738-746, 767-815, 832-879, 888-928, 939-979, 992-1004, 1016-1028, 1044-1070, 1086-1112, 1124-1149, 1161-1186, 1199-1223, 1240-1264, 1282-1310, 1328-1356, 1399-1424, 1439-1464, 1480-1497, 1511-1528, 1563-1605, 1637-1665, 1711-1715, 1721, 1775-1778, 1814-1858, 1890-1919, 1964-1968, 1974-1999, 2024, 2030-2033, 2048-2054, 2063-2068, 2083-2088, 2242, 2285, 2309-2320, 2337-2348, 2355-2369, 2423, 2426, 2457-2463, 2471-2487, 2494-2497, 2516-2517, 2553-2559, 2562-2563, 2579-\>exit |
| src/haystack\_integrations/document\_stores/qdrant/filters.py         |      125 |      102 |       60 |        2 |     14% |20, 24-29, 34-54, 59-80, 85-93, 98-109, 114-130, 134-136, 140-143, 156, 168-171, 184-192, 196-204, 208-216, 220-228, 233-237 |
| **TOTAL**                                                             | **1079** |  **697** |  **340** |   **25** | **30%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/python-coverage-comment-action-data-qdrant/endpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdeepset-ai%2Fhaystack-core-integrations%2Fpython-coverage-comment-action-data-qdrant%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack-core-integrations/blob/python-coverage-comment-action-data-qdrant/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.