# Comet API Haystack Integration

[![PyPI - Version](https://img.shields.io/pypi/v/cometapi-haystack.svg)](https://pypi.org/project/cometapi-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cometapi-haystack.svg)](https://pypi.org/project/cometapi-haystack)

-----

**CometAPI Resources**
- [Website](https://www.cometapi.com/?utm_source=haystack&utm_campaign=integration&utm_medium=integration&utm_content=integration)
- [Documentation](https://api.cometapi.com/doc)
- [Get an API Key](https://api.cometapi.com/console/token)
- [Pricing](https://api.cometapi.com/pricing)

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install cometapi-haystack
```

## Usage

This integration offers a set of pre-built components that allow developers to interact seamlessly with AI models using the new Comet APIs.

### Chat Generator

```python
from haystack.dataclasses.chat_message import ChatMessage
from haystack_integrations.components.generators.cometapi import CometAPIChatGenerator

# Initialize the chat generator
chat_generator = CometAPIChatGenerator(model="grok-3-mini")

# Generate a response
messages = [ChatMessage.from_user("Tell me about the future of AI")]
response = chat_generator.run(messages=messages)
print(response["replies"][0].text)
```


## License

`cometapi-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. 