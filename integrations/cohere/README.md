# cohere-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/cohere-haystack.svg)](https://pypi.org/project/cohere-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cohere-haystack.svg)](https://pypi.org/project/cohere-haystack)

-----

**Table of Contents**

- [cohere-haystack](#cohere-haystack)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

```console
pip install cohere-haystack
```

## Contributing

`hatch` is the best way to interact with this project, to install it:
```sh
pip install hatch
```

With `hatch` installed, to run all the tests:
```
hatch run test
```
> Note: integration tests will be skipped unless the env var COHERE_API_KEY is set. The api key needs to be valid
> in order to pass the tests.

To only run unit tests:
```
hatch run test -m"not integration"
```

To only run embedders tests:
```
hatch run test -m"embedders"
```

To only run generators tests:
```
hatch run test -m"generators"
```

Markers can be combined, for example you can run only integration tests for embedders with:
```
hatch run test -m"integrations and embedders"
```

To run the linters `ruff` and `mypy`:
```
hatch run lint:all
```

## License

`cohere-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
