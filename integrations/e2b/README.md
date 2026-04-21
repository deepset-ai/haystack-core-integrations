# e2b-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/e2b-haystack.svg)](https://pypi.org/project/e2b-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/e2b-haystack.svg)](https://pypi.org/project/e2b-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/e2b/CHANGELOG.md)

---

**e2b-haystack** provides [E2B](https://e2b.dev/) cloud sandbox tools for
[Haystack](https://haystack.deepset.ai/) agents. It exposes four tools that
operate inside a shared sandbox environment:

| Tool | Description |
|------|-------------|
| `RunBashCommandTool` | Execute bash commands |
| `ReadFileTool` | Read file contents |
| `WriteFileTool` | Write files |
| `ListDirectoryTool` | List directory contents |

All tools share a single `E2BSandbox` instance so the agent can write a file in
one step and read or execute it in the next.

## Installation

```bash
pip install e2b-haystack
```

## Usage

Set the `E2B_API_KEY` environment variable (get one at <https://e2b.dev/>).

### Quick start with `E2BToolset`

The simplest way to use all four tools together:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

from haystack_integrations.tools.e2b import E2BToolset

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=E2BToolset(),
)
```

### Using individual tools

For more control, create an `E2BSandbox` and pass it to the tools you need:

```python
from haystack_integrations.tools.e2b import (
    E2BSandbox,
    RunBashCommandTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

sandbox = E2BSandbox()
tools = [
    RunBashCommandTool(sandbox=sandbox),
    ReadFileTool(sandbox=sandbox),
    WriteFileTool(sandbox=sandbox),
    ListDirectoryTool(sandbox=sandbox),
]
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
