# amazon-sagemaker-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/amazon-sagemaker-haystack.svg)](https://pypi.org/project/amazon-sagemaker-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/amazon-sagemaker-haystack.svg)](https://pypi.org/project/amazon-sagemaker-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Testing](#testing)
- [License](#license)

## Installation

```console
pip install amazon-sagemaker-haystack
```

## Testing

To run the test suite locally, run:

```console
hatch run test
```

You need to also export your AWS credentials for Sagemaker integration tests to run (`AWS_ACCESS_KEY_ID` and 
`AWS_SECRET_SECRET_KEY`). If those are missing, the tests will be skipped.

## License

`amazon-sagemaker-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
