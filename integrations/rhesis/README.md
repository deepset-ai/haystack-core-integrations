# rhesis-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/rhesis-haystack.svg)](https://pypi.org/project/rhesis-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rhesis-haystack.svg)](https://pypi.org/project/rhesis-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/rhesis/CHANGELOG.md)

---

## Installation

```bash
pip install rhesis-haystack
```

## Required environment variables

Set these before running your application:

| Variable | Required | Description |
|---|---|---|
| `RHESIS_API_KEY` | Yes | API key for trace ingestion |
| `RHESIS_BASE_URL` | No | Backend URL (default `http://localhost:8080`) |
| `RHESIS_PROJECT_ID` | No | Project ID (resolved from API key when omitted) |
| `RHESIS_ENVIRONMENT` | No | Environment label (default `development`) |
| `RHESIS_FRONTEND_URL` | No | Frontend URL for `trace_url` deep links |
| `HAYSTACK_CONTENT_TRACING_ENABLED` | Yes | Must be `"true"` **before importing Haystack** |
| `HAYSTACK_RHESIS_ENFORCE_FLUSH` | No | Default `"true"`; exports once per run. See [Flush behavior](#flush-behavior) |

> **Important:** Set `HAYSTACK_CONTENT_TRACING_ENABLED=true` before any `haystack` import, otherwise
> input/output content tags are no-ops. Haystack reads this variable once, when `haystack.tracing`
> is first imported, so setting it afterwards has no effect.

## Quickstart

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.rhesis import RhesisConnector

pipe = Pipeline()
pipe.add_component("tracer", RhesisConnector("Chat example"))
pipe.add_component("prompt_builder", ChatPromptBuilder())
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
pipe.connect("prompt_builder.prompt", "llm.messages")

messages = [
    ChatMessage.from_system("Always respond in German."),
    ChatMessage.from_user("Tell me about {{location}}"),
]

response = pipe.run(
    data={
        "prompt_builder": {
            "template_variables": {"location": "Berlin"},
            "template": messages,
        },
        "tracer": {"invocation_context": {"session_id": "demo-session"}},
    }
)
print(response["llm"]["replies"][0])
print(response["tracer"]["trace_url"])
print(response["tracer"]["trace_id"])
```

## Agents, and tracing without a pipeline

Constructing `RhesisConnector` is what installs the tracer, so a standalone `Agent` needs nothing
else — build one and never mention it again. There is no pipeline to carry the `invocation_context`
input socket, so attach session and test metadata with `rhesis_invocation_context` instead:

```python
from haystack_integrations.components.connectors.rhesis import RhesisConnector
from haystack_integrations.tracing.rhesis import rhesis_invocation_context

RhesisConnector("Agent example")  # installs the tracer; never added to a pipeline

with rhesis_invocation_context({"session_id": "agent-example", "test_run_id": "tr-1"}):
    result = agent.run(messages=[ChatMessage.from_user("What is the weather in Berlin?")])
```

Every span opened inside the block joins that session, and the context is restored on exit.

The same context manager works around a `pipeline.run()` call, and there it does something the
input socket cannot. Both attach the context to the run's root span, which is what conversation
grouping and the `trace_url` need. But the socket supplies its value from *inside* the run, so a
component whose span closed before the connector executed has already been exported without it.
Wrapping the call instead means no span opens without the context:

```python
with rhesis_invocation_context({"session_id": "chat-42", "test_run_id": "tr-1"}):
    result = pipe.run({"prompt_builder": {...}})
```

Use the socket when you only need the run grouped; use the context manager when you want to filter
a trace's individual spans by your own ids.

Keys the mapping knows — `session_id`, `conversation_id`, `test_run_id`, `test_id`,
`test_result_id`, `test_configuration_id` — become first-class Rhesis attributes. Anything else,
`user_id` and `tags` included, travels as `haystack.invocation.<key>`.

See [`example/agent.py`](./example/agent.py) for a runnable version with two tools.

## Conversations

`RhesisConnector` traces each pipeline run. An application that owns its own loop — a chat REPL, a
batch script, a server handling one turn per request — needs two more things a component inside the
pipeline cannot give it: tracing switched on without a pipeline to attach to, and a span wrapping a
whole pipeline run so a conversation turn has a root of its own.

Without that root, the pipeline span claims the turn and reports the serialized pipeline input and
output as the conversation text.

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack_integrations.tracing.rhesis import RhesisTracing

tracing = RhesisTracing("My Assistant")  # no-op when RHESIS_API_KEY is unset
tracing.start_conversation("conversation-1")

for message in ["I have a headache", "It started three days ago"]:
    with tracing.turn(message) as turn:
        result = pipeline.run(...)
        turn.output = extract_reply(result)  # what the user actually sees

tracing.flush()
```

Every turn after the first joins the first one's trace, so a conversation reads as one trace rather
than one per exchange. Call `start_conversation` again to begin a new one.

Assign `turn.output` yourself: only the application knows which part of a pipeline result is the
reply — it may be a tool result, or a value held in agent state rather than the last assistant
message.

Pass `enabled=False` to build a no-op instance when your own configuration says tracing should be
off, and `turn_span_name=...` to name the turn spans after your application.

## Flush behavior

By default (`HAYSTACK_RHESIS_ENFORCE_FLUSH=true`) the tracer exports once per pipeline run, when
the root span closes, so everything the run produced is on the backend by the time `run()` returns.
That costs one blocking round trip per run.

Set it to `false` to hand exporting to the SDK's `BatchSpanProcessor` instead and pay nothing on the
request path. Spans are then batched and sent in the background, and OpenTelemetry's `atexit` hook
flushes what is left when the process exits normally — so a short script loses nothing either way.

Turn it off when the round trip matters and normal exit is guaranteed; leave it on when it is not:

- the process can be hard-killed (`os._exit`, `SIGKILL`, a container OOM kill),
- a serverless runtime freezes the sandbox after the response is returned,
- you cannot call `flush()` yourself at shutdown.

```python
import os

os.environ["HAYSTACK_RHESIS_ENFORCE_FLUSH"] = "false"

from haystack.tracing import tracer

try:
    ...
finally:
    tracer.actual_tracer.flush()
```

## Custom SpanHandler

```python
from haystack_integrations.tracing.rhesis import DefaultSpanHandler, RhesisSpan


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span: RhesisSpan, component_type: str | None) -> None:
        super().handle(span, component_type)
        # add custom attributes here


connector = RhesisConnector("My app", span_handler=CustomSpanHandler())
```

## Semantic mapping

| Haystack source | Rhesis target | Promotion |
|---|---|---|
| `haystack.pipeline.run` | `function.haystack.pipeline.run` | First-class span name |
| `haystack.async_pipeline.run` | `function.haystack.async_pipeline.run` | First-class span name |
| `haystack.agent.run` (root) | `ai.agent.invoke` | First-class |
| `*ChatGenerator` / `*Generator` | `ai.llm.invoke` + model/tokens | First-class |
| `*Retriever` | `ai.retrieval` | First-class |
| `*Embedder` | `ai.embedding.generate` | First-class |
| `ToolInvoker` | `ai.tool.invoke` | First-class |
| `haystack.component.input/output` | `ai.prompt` / `ai.completion` events | First-class (content-gated) |
| `haystack.pipeline.input_data/output_data` | `rhesis.conversation.input/output` | First-class (content-gated) |
| `invocation_context.session_id` | `rhesis.conversation.id` | First-class |
| Other Haystack tags | `haystack.*` metadata | Metadata |

See [`mapping.py`](./src/haystack_integrations/tracing/rhesis/mapping.py) for the full, authoritative mapping table.

## Limitations

What this integration does and does not do, so none of it arrives as a surprise:

- **The content flag is read once, at Haystack import.** Setting
  `HAYSTACK_CONTENT_TRACING_ENABLED` after `haystack.tracing` has been imported has no effect, and
  the integration cannot detect the mistake — by the time `RhesisConnector` is importable the value
  has already been read. It warns at construction when the flag is off.
- **Content tracing gates our attributes, not Haystack's tags.** With the flag off, no `ai.prompt`
  or `ai.completion` events are recorded and nothing is promoted to `rhesis.conversation.input` /
  `.output`. Haystack still passes pipeline I/O as an ordinary span tag, so
  `haystack.pipeline.input_data` carries the run payload either way — the same behaviour as the
  langfuse integration. Truncated to 8 000 characters.
- **Content leaves the process verbatim.** With the flag on, prompts, tool arguments and retrieved
  documents are sent to Rhesis as written. There is no redaction hook; use a custom `SpanHandler` if
  you need one.
- **Conversation grouping needs a `session_id`.** Without one a run is traced but not grouped. Turns
  share a trace only when driven through `RhesisTracing`.
- **`ComponentTool` invocations are flattened.** The tool span is recorded; the component it wraps
  gets no span of its own, so the tree does not nest there.
- **Construction fails closed on a missing API key.** `RhesisConnector(...)` raises `ValueError` when
  no key resolves, so `Pipeline.from_dict` on a YAML containing it needs credentials present. An
  invalid key only logs — it is rejected at export time, not construction.
- **Truncation is silent.** Conversation attributes are capped at 10 000 characters and content
  events at 8 000; nothing marks a value as truncated.

## Local development

Point at a local Rhesis backend:

```bash
export RHESIS_API_KEY=your-key
export RHESIS_BASE_URL=http://localhost:8080
export RHESIS_FRONTEND_URL=http://localhost:3000
export HAYSTACK_CONTENT_TRACING_ENABLED=true
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally:

```bash
cd integrations/rhesis
export RHESIS_API_KEY=your-key
export RHESIS_BASE_URL=http://localhost:8080  # optional
hatch run test:integration
```
