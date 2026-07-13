# plivo-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/plivo-haystack.svg)](https://pypi.org/project/plivo-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/plivo-haystack.svg)](https://pypi.org/project/plivo-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/plivo/CHANGELOG.md)

---

Haystack tools that let an [Agent](https://docs.haystack.deepset.ai/docs/agent) use the
[Plivo](https://www.plivo.com/) CPaaS APIs: send SMS, run Verify (OTP) send/validate,
look up a number's carrier, and place an outbound call.

| Tool | Plivo API | Action |
|---|---|---|
| `SendSMSTool` | Messages | Send an SMS from the account number |
| `SendVerificationTool` | Verify | Send a one-time code over SMS or voice |
| `ValidateVerificationTool` | Verify | Validate a code against its session |
| `LookupNumberTool` | Lookup | Carrier / line-type / country lookup |
| `MakeCallTool` | Voice | Place an outbound call (fetches your answer XML) |

`MakeCallTool` only *initiates* a call; Haystack has no telephony/audio runtime, so
real-time voice (audio streaming) is out of scope for this integration.

## Installation

```console
pip install plivo-haystack
```

## Usage

Set your Plivo credentials (find them in the console at <https://cx.plivo.com/>):

```console
export PLIVO_AUTH_ID="..."
export PLIVO_AUTH_TOKEN="..."
```

Give the whole toolset to an Agent:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

from haystack_integrations.tools.plivo import PlivoToolset

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=PlivoToolset(sender="+14150000000"),
)
result = agent.run(messages=[...])
```

Or pick individual tools that share one client:

```python
from haystack_integrations.tools.plivo import PlivoClient, SendSMSTool, LookupNumberTool

client = PlivoClient(sender="+14150000000")
tools = [SendSMSTool(client=client), LookupNumberTool(client=client)]
```

`sender` is required only by `SendSMSTool` and `MakeCallTool`.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, set `PLIVO_AUTH_ID` and `PLIVO_AUTH_TOKEN` (get them at <https://cx.plivo.com/>).
