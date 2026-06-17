# tika-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/tika-haystack.svg)](https://pypi.org/project/tika-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tika-haystack.svg)](https://pypi.org/project/tika-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/tika/CHANGELOG.md)

---

Haystack integration for [Apache Tika](https://tika.apache.org/), providing the `TikaDocumentConverter` component
that converts files of different types (PDF, DOCX, HTML, and more) to Haystack Documents.

## Installation

```shell
pip install tika-haystack
```

## Running Tika

This integration requires a running Tika server. The easiest way is via Docker:

```shell
docker run -d -p 127.0.0.1:9998:9998 apache/tika:latest
```

For more options, see the [Tika Docker documentation](https://github.com/apache/tika-docker/blob/main/README.md#usage).

## Usage

```python
from haystack_integrations.components.converters.tika import TikaDocumentConverter
from pathlib import Path

converter = TikaDocumentConverter()
result = converter.run(sources=[Path("my_file.pdf")])
documents = result["documents"]
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Integration tests require a running Tika server. Start one with:

```shell
docker run -d -p 127.0.0.1:9998:9998 apache/tika:latest
```
