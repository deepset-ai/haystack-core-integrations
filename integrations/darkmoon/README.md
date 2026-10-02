# darkmoon-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/darkmoon-haystack.svg)](https://pypi.org/project/darkmoon-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/darkmoon-haystack.svg)](https://pypi.org/project/darkmoon-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/darkmoon)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/darkmoon/CHANGELOG.md)

---

`DarkmoonToolset` lets a Haystack `Agent` drive a self-hosted [Darkmoon](https://github.com/ASCIT31/Dark-Moon), a GPL-3.0 autonomous AI penetration testing platform. It exposes three tools: `darkmoon_run_pentest`, `darkmoon_get_findings` and `darkmoon_list_campaigns`.

The tools call the Darkmoon Dashboard API of an instance you operate. The Darkmoon engine and CLI are open source; the Dashboard API used here is part of Darkmoon's Pro edition, and the Pro remediation-to-pull-request feature is not exposed. Only test systems you own or are authorised to test, and review findings for false positives.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.darkmoon import DarkmoonToolset

# Requires DARKMOON_BASE_URL, DARKMOON_USERNAME, DARKMOON_PASSWORD and OPENAI_API_KEY
agent = Agent(chat_generator=OpenAIChatGenerator(), tools=DarkmoonToolset())
result = agent.run(messages=[ChatMessage.from_user("List my Darkmoon campaigns and summarise the latest one.")])
print(result["last_message"].text)
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The unit tests mock the Darkmoon Dashboard API and need no credentials or running instance.
