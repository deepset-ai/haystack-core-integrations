# ollama-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/ollama-haystack.svg)](https://pypi.org/project/ollama-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ollama-haystack.svg)](https://pypi.org/project/ollama-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install ollama-haystack
```

## License

`ollama-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## Testing

To run tests first start a Docker container running Ollama. We provide a utility `docker-compose.yml` for that:

```console
docker-compose up -d
```

Then run tests:

```console
hatch run test
```

The default model used here is ``orca-mini``