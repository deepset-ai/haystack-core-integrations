# ag2-haystack

[![PyPI](https://img.shields.io/pypi/v/ag2-haystack.svg)](https://pypi.org/project/ag2-haystack/)

An integration of [AG2](https://ag2.ai/) (formerly AutoGen) with [Haystack](https://haystack.deepset.ai/).

AG2 is a multi-agent conversation framework with 500K+ monthly PyPI downloads, 4,300+ GitHub stars, and 400+ contributors. This integration brings AG2's powerful multi-agent orchestration capabilities into Haystack pipelines.

## Installation

```bash
pip install ag2-haystack
```

## Usage

### Standalone

```python
import os
from haystack_integrations.components.agents.ag2 import AG2Agent

os.environ["OPENAI_API_KEY"] = "your-key"

agent = AG2Agent(
    model="gpt-4o-mini",
    system_message="You are a helpful research assistant.",
)

result = agent.run(query="What are the latest advances in RAG?")
print(result["reply"])
```

### In a Haystack Pipeline

```python
from haystack import Pipeline
from haystack_integrations.components.agents.ag2 import AG2Agent

pipeline = Pipeline()
pipeline.add_component("agent", AG2Agent(
    model="gpt-4o-mini",
    system_message="Answer questions clearly and concisely.",
))

result = pipeline.run({"agent": {"query": "Explain retrieval-augmented generation."}})
print(result["agent"]["reply"])
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"gpt-4o-mini"` | LLM model name |
| `system_message` | str | `"You are a helpful AI assistant."` | System message for the assistant |
| `api_key_env_var` | str | `"OPENAI_API_KEY"` | Env var name for the API key |
| `api_type` | str | `"openai"` | API type (`"openai"`, `"bedrock"`, etc.) |
| `max_consecutive_auto_reply` | int | `10` | Max auto-replies |
| `human_input_mode` | str | `"NEVER"` | Human input mode |
| `code_execution` | bool | `False` | Enable code execution |

## License

Apache-2.0 — See [LICENSE](../../LICENSE) for details.
