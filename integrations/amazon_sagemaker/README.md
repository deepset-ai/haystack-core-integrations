# amazon-sagemaker-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/amazon-sagemaker-haystack.svg)](https://pypi.org/project/amazon-sagemaker-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/amazon-sagemaker-haystack.svg)](https://pypi.org/project/amazon-sagemaker-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Installation

```console
pip install amazon-sagemaker-haystack
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

> Note: You need to export your AWS credentials for Sagemaker integration tests to run (`AWS_ACCESS_KEY_ID` and 
`AWS_SECRET_SECRET_KEY`). If those are missing, the integration tests will be skipped.

To only run unit tests:
```
hatch run test:unit
```

To only run integration tests:
```
hatch run test:integration
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

`amazon-sagemaker-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
