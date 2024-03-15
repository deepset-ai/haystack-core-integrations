# anthropic-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/anthropic-haystack.svg)](https://pypi.org/project/anthropic-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anthropic-haystack.svg)](https://pypi.org/project/anthropic-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Contributing](#contributing)
- [Examples](#examples)
- [License](#license)

## Installation

```console
pip install anthropic-haystack
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
> Note: there are no integration tests for this project.

To run the linters `ruff` and `mypy`:
```
hatch run lint:all
```

## Examples
You can find an example of how to do a simple RAG with Claude using online documentation in the `example/` folder of this repo.

## License

`anthropic-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
