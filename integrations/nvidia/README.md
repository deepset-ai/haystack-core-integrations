# nvidia-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/nvidia-haystack.svg)](https://pypi.org/project/nvidia-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nvidia-haystack.svg)](https://pypi.org/project/nvidia-haystack)

---

**Table of Contents**

- [nvidia-haystack](#nvidia-haystack)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

```console
pip install nvidia-haystack
```

## Contributing

`hatch` is the best way to interact with this project, to install it:

```sh
pip install hatch
```

With `hatch` installed, to run all the tests:

```
hatch run test:all
```

> Note: integration tests will be skipped unless the env var NVIDIA_API_KEY is set. The api key needs to be valid
> in order to pass the tests.

To only run unit tests:

```
hatch run test:unit
```

To format your code and perform linting using Ruff (with automatic fixes), run:

```
hatch run fmt
```

To check for static type errors, run:

```console
$ hatch run test:types
```

## License

`nvidia-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
