# ag2-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/ag2-haystack.svg)](https://pypi.org/project/ag2-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ag2-haystack.svg)](https://pypi.org/project/ag2-haystack)

An integration between [AG2](https://ag2.ai/) (formerly AutoGen) multi-agent framework and [Haystack](https://haystack.deepset.ai/).

## Installation

```console
pip install ag2-haystack
```

## Usage

```python
import os
from haystack import Pipeline
from haystack_integrations.components.agents.ag2 import AG2Agent

os.environ["OPENAI_API_KEY"] = "your-api-key"

pipeline = Pipeline()
pipeline.add_component("agent", AG2Agent(model="gpt-4o-mini"))

result = pipeline.run({"agent": {"query": "Explain RAG in one sentence."}})
print(result["agent"]["reply"])
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | Model identifier |
| `api_type` | `str` | `"openai"` | API type (e.g. `"openai"`, `"azure"`) |
| `system_message` | `str \| None` | `None` | Optional system message for the assistant |
| `human_input_mode` | `str` | `"NEVER"` | One of `"NEVER"`, `"TERMINATE"`, `"ALWAYS"` |
| `code_execution` | `bool` | `False` | Enable code execution in UserProxyAgent |
| `max_consecutive_auto_reply` | `int` | `10` | Max auto-replies before stopping |

## License

`ag2-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
