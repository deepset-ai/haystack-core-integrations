# iflytek-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/iflytek-haystack.svg)](https://pypi.org/project/iflytek-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/iflytek-haystack.svg)](https://pypi.org/project/iflytek-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/iflytek/CHANGELOG.md)

---

An integration between [iFlytek Spark](https://www.xfyun.cn/) and [Haystack](https://haystack.deepset.ai/). iFlytek Spark exposes an OpenAI-compatible chat completion API, so `IFlytekChatGenerator` builds on Haystack's `OpenAIChatGenerator`.

## Installation

```console
pip install iflytek-haystack
```

## Usage

Get an API password from the [iFlytek open platform console](https://console.xfyun.cn/) and set it as the `IFLYTEK_API_KEY` environment variable, then pick a model such as `generalv3.5`, `4.0Ultra` or `lite`.

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.iflytek import IFlytekChatGenerator

client = IFlytekChatGenerator(model="4.0Ultra")
response = client.run([ChatMessage.from_user("用一句话介绍你自己")])
print(response)
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
