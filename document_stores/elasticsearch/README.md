[![test](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/document_stores_elasticsearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/document_stores_elasticsearch.yml)

[![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)

# Elasticsearch Document Store

Document Store for Haystack 2.x, supports ElasticSearch 8.

## Installation

```console
pip install elasticsearch-haystack
```

## Testing

To run tests first start a Docker container running ElasticSearch. We provide a utility `docker-compose.yml` for that:

```console
docker-compose up
```

Then run tests:

```console
hatch run test
```

## License

`elasticsearch-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
