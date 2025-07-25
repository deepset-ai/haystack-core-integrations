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

To run tests first start a Docker container running Ollama and pull a model for integration testing
It's recommended to use the smallest model possible for testing purposes - see https://ollama.ai/library for a list that Ollama supportd

```console
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
docker exec ollama ollama pull <your model here>
```

Then run tests:

```console
hatch run test:all
```

The default model used here is ``orca-mini`` for generation and ``nomic-embed-text`` for embeddings