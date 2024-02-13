# pgvector-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/pgvector-haystack.svg)](https://pypi.org/project/pgvector-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pgvector-haystack.svg)](https://pypi.org/project/pgvector-haystack)

---

**Table of Contents**

- [pgvector-haystack](#pgvector-haystack)
  - [Installation](#installation)
  - [Testing](#testing)
  - [License](#license)

## Installation

```console
pip install pgvector-haystack
```

## Testing

Ensure you have PostgreSQL installed with the `pgvector` extension, for a quick setup using Docker:
```
docker run -d -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres ankane/pgvector
```


```console
hatch run test
```

## License

`pgvector-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
