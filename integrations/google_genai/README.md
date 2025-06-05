# Google Gen AI Haystack Integration

[![PyPI - Version](https://img.shields.io/pypi/v/google-genai-haystack.svg)](https://pypi.org/project/google-genai-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/google-genai-haystack.svg)](https://pypi.org/project/google-genai-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install google-genai-haystack
```

## Usage

This integration provides components to use Google's Gemini models via the new Google Gen AI SDK.

### Chat Generator

```python
from haystack.dataclasses.chat_message import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Initialize the chat generator
chat_generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash")

# Generate a response
messages = [ChatMessage.from_user("Tell me about the future of AI")]
response = chat_generator.run(messages=messages)
print(response["replies"][0].text)
```

### Streaming Chat Generator

```python
from haystack.dataclasses.chat_message import ChatMessage
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

def streaming_callback(chunk: StreamingChunk):
    print(chunk.content, end='', flush=True)

# Initialize with streaming callback
chat_generator = GoogleGenAIChatGenerator(
    model="gemini-2.0-flash",
    streaming_callback=streaming_callback
)

# Generate a streaming response
messages = [ChatMessage.from_user("Write a short story")]
response = chat_generator.run(messages=messages)
# Text will stream in real-time via the callback
```

## License

`google-genai-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. 