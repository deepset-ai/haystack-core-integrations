# dbos-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dbos-haystack.svg)](https://pypi.org/project/dbos-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dbos-haystack.svg)](https://pypi.org/project/dbos-haystack)

Durable execution for Haystack Agents, powered by [DBOS](https://docs.dbos.dev).

DBOS checkpoints each step of a run to Postgres or SQLite. When a process crashes and restarts, DBOS
re-executes the unfinished workflow and replays every completed step from its recorded output, so an Agent
resumes at the last completed model call instead of starting over. It runs in-process as a library — there
is no external orchestrator to deploy.

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/dbos/CHANGELOG.md)

---

## Installation

```console
pip install dbos-haystack
```

## Usage

Wrap the Agent's chat generator with `durable_agent`, then call `agent.run()` from inside your own
`@DBOS.workflow()`:

```python
from dbos import DBOS, DBOSConfig
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.dbos import durable_agent

agent = durable_agent(
    Agent(chat_generator=OpenAIChatGenerator(), tools=[...]),
    name="support_agent",
)


@DBOS.workflow()
def answer(question: str) -> str:
    result = agent.run(messages=[ChatMessage.from_user(question)])
    return result["last_message"].text


config: DBOSConfig = {"name": "support", "system_database_url": "sqlite:///dbos.sqlite"}
DBOS(config=config)
DBOS.launch()
```

Use Postgres in production by pointing `system_database_url` at it. Outside a DBOS workflow — in tests, or in
an application that has not called `DBOS.launch()` — the wrapped generator behaves exactly like the generator
it wraps, so the same Agent works either way.

Declare your workflows at module scope, before `DBOS.launch()` — DBOS needs them registered to recover them after
a restart. The Agent itself can be built whenever you like.

A runnable end-to-end example, including what a recovery looks like, is in
[`examples/durable_agent.py`](examples/durable_agent.py).

## Durable human-in-the-loop

`DBOSConfirmationStrategy` plugs into Haystack's stock `ConfirmationHook`. Instead of blocking on console
input, it publishes the pending tool call as a DBOS event and suspends the workflow on `DBOS.recv` — so the
run can wait hours or days for an approval and survive a restart while it waits.

```python
from haystack.hooks.human_in_the_loop import ConfirmationHook
from haystack_integrations.dbos import DBOSConfirmationStrategy, durable_agent

hook = ConfirmationHook(confirmation_strategies={"delete_file": DBOSConfirmationStrategy()})
agent = durable_agent(
    Agent(chat_generator=OpenAIChatGenerator(), tools=[delete_file], hooks={"before_tool": [hook]}),
    name="support_agent",
)
```

Approve or reject from anywhere — an HTTP handler, a Slack bot, a CLI:

```python
pending = DBOS.get_event(workflow_id, "haystack.pending_tool_call", timeout_seconds=0)
DBOS.send(workflow_id, {"action": "confirm"}, topic="confirm.delete_file")
```

Because `set_event` and `recv` are themselves checkpointed, a recovered workflow re-reads the decision that
was already given rather than asking again.

## Durability guarantees

|                           | Checkpointed | On recovery                                                                                                            |
| ------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Model calls               | yes          | Replayed from the database, not re-issued                                                                                |
| Tool calls                | no           | **Re-execute.** Make them idempotent, or decorate the tool function with `@DBOS.step` *and* use `run_async` (or `tool_concurrency_limit=1`) |
| Human confirmations       | yes          | Replayed — the human is not asked twice                                                                                  |
| Token streaming           | no           | No chunks are emitted for already-completed model calls                                                                  |
| Tool-result stream chunks | no           | Re-fire for the replayed prefix                                                                                          |
| Tracing spans             | no           | Re-created — expect duplicate spans in Langfuse/OpenTelemetry                                                             |
| Dynamic toolsets          | —            | **Unsupported.** Use static tool lists                                                                                   |

### Why tool calls re-execute

Recovery is replay, not forward-resume: the workflow body runs again from the top and only completed steps are
short-circuited. Anything that is not a step therefore runs a second time. This is the standard durable-execution
contract, and it is why tools must either be idempotent or be steps themselves.

To make a tool durable, decorate its function with `@DBOS.step()` and run the Agent with `run_async`:

```python
@tool
@DBOS.step()
def charge_card(customer_id: str, amount: int) -> str: ...
```

Use `Agent.run_async()` for this, or set `tool_concurrency_limit=1`. Synchronous `Agent.run()` executes tools in
a `ThreadPoolExecutor` that shares DBOS's step-id counter across workers, so with concurrency above 1 two
stepped tool calls can be assigned the same id — which either fails the recovery outright or, when two calls to
the same tool collide, returns one call's cached result for the other. DBOS states the rule directly: *"You
should not use threads to start workflows or to start steps in workflows."* The async path is safe: tool
coroutines are created in call order and the semaphore wakes waiters FIFO.

### Other caveats

- **Dynamic toolsets are unsupported.** A toolset that discovers tools as a side effect of running one of its
  tools (such as `SearchableToolset`) will not re-populate on replay, because the discovery step returns its
  recorded output without running its body. Use static tool lists.
- **Nested agents lose durability.** An Agent invoked from inside a step is not itself checkpointed, because
  DBOS does not treat a step as a workflow context.
- **Do not mix sync and async.** Call a synchronous `Agent.run()` from a synchronous workflow and
  `run_async()` from an async workflow.
- **Pin `haystack-ai`.** The number and order of model calls per Agent step is part of the replay contract; an
  upstream change to the loop can strand workflows that were in flight across the upgrade. DBOS only recovers
  workflows whose `application_version` matches, so those workflows are stranded for manual `fork_workflow`
  rather than corrupted.

## Development

```console
cd integrations/dbos
hatch run fmt
hatch run test:types
hatch run test:unit
```

The test suite runs against in-process SQLite and needs no database service.

## License

`dbos-haystack` is distributed under the terms of the
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
