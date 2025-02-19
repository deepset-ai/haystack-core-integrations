# weights_biases-haystack

[![PyPI - License](https://img.shields.io/pypi/l/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)
[![PyPI - Version](https://img.shields.io/pypi/v/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/weights_bias-haystack.svg)](https://pypi.org/project/weights_bias-haystack)

---

**Table of Contents**

- [weights_bias-haystack](#weights_bias-haystack)
  - [Installation](#installation)
  - [Example](#example)
  - [License](#license)


## Installation

```console
pip install weights_biases-haystack
```

## Weave by Weights & Biases

### Example 

You need to have a Weave account to use this feature. You can sign up for free at https://wandb.ai/site.

You then need to set the `WANDB_API_KEY:` environment variable with your Weights & Biases API key. You should find 
your API key in https://wandb.ai/home when you are logged in. 

You should then head to `https://wandb.ai/<user_name>/projects` and see the complete trace for your pipeline under
the pipeline name you specified, when creating the `WeaveConnector`.

You also need to have the `HAYSTACK_CONTENT_TRACING_ENABLED` environment variable set to `true`.

To use this connector simply add it to your pipeline without any connections, and it will automatically start 
sending traces to Weights & Biases.

```python
from haystack.pipeline import WeaveConnector
