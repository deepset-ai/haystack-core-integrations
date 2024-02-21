# optimum

[![PyPI - Version](https://img.shields.io/pypi/v/optimum.svg)](https://pypi.org/project/optimum-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/optimum.svg)](https://pypi.org/project/optimum-haystack)

-----

Component to embed strings and Documents using models loaded with the HuggingFace Optimum library. This component is designed to seamlessly inference models using the high speed ONNX runtime.

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

To use the ONNX runtime for CPU, use the CPU version:
```console
pip install optimum-haystack[cpu]
```

For using the GPU runtimes:
```console
pip install optimum-haystack[gpu]
```


## License

`optimum-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
