# AIMLAPI-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/aimlapi-haystack.svg)](https://pypi.org/project/aimlapi-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aimlapi-haystack.svg)](https://pypi.org/project/aimlapi-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/aimlapi)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/aimlapi/CHANGELOG.md)

---

## Installation

```bash
pip install aimlapi-haystack
```

## Usage

```python
from haystack_integrations.components.generators.aimlapi import AIMLAPIChatGenerator
from haystack.dataclasses import ChatMessage

generator = AIMLAPIChatGenerator(model="openai/gpt-5-chat-latest")
result = generator.run([ChatMessage.from_user("What's the capital of France?")])
print(result["replies"][0].content)
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `AIMLAPI_API_KEY` environment variable.
