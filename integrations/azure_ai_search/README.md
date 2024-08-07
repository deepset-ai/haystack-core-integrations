[![test](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml/badge.svg)](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/opensearch.yml)

[![PyPI - Version](https://img.shields.io/pypi/v/opensearch-haystack.svg)](https://pypi.org/project/opensearch-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opensearch-haystack.svg)](https://pypi.org/project/opensearch-haystack)

# OpenSearch Document Store

Document Store for Haystack 2.x, supports OpenSearch.

## Installation

```console
pip install opensearch-haystack
```

## Testing

To run tests first start a Docker container running OpenSearch. We provide a utility `docker-compose.yml` for that:

```console
docker-compose up
```

Then run tests:

```console
hatch run test
```

## License

`opensearch-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
