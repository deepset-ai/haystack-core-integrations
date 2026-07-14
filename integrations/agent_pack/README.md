# agent-pack-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/agent_pack/CHANGELOG.md)

---

**agent-pack-haystack** is a pack of ready-to-use agents built on Haystack. Each agent is exposed as
a factory that returns a configured Haystack `Agent`.

## Agents

- **Deep Research Agent** — give it a question, and it researches the web and produces a structured,
  cited markdown report. See its [README](src/haystack_integrations/components/agents/agent_pack/deep_research/README.md)
  for the architecture and full configuration.

## Installation

```bash
pip install agent-pack-haystack
```

The deep research agent has optional runtime dependencies for web search and HTML/PDF parsing.
Install them alongside the package to use it:

```bash
pip install agent-pack-haystack tavily-haystack trafilatura pypdf arrow
```

## Usage

```python
from haystack.dataclasses import ChatMessage
from agent_pack_haystack import create_deep_research_agent

agent = create_deep_research_agent()
result = agent.run(messages=[ChatMessage.from_user("your research question")])
print(result["report"])
```

The same factory is also importable from its canonical namespace location,
`haystack_integrations.components.agents.agent_pack`.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
