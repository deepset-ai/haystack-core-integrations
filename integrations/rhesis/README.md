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
| `HAYSTACK_RHESIS_ENFORCE_FLUSH` | No | Default `"true"`; set `"false"` in long-running services |

> **Important:** Set `HAYSTACK_CONTENT_TRACING_ENABLED=true` before any `haystack` import, otherwise
> input/output content tags are no-ops.

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

## Flush behavior

By default spans are flushed after each component (`HAYSTACK_RHESIS_ENFORCE_FLUSH=true`). For
long-running services, disable per-span flush and flush on shutdown:

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
| `haystack.pipeline.input_data/output_data` | `rhesis.conversation.input/output` | First-class |
| `invocation_context.session_id` | `rhesis.conversation.id` | First-class |
| Other Haystack tags | `haystack.*` metadata | Metadata |

See [`mapping.py`](./src/haystack_integrations/tracing/rhesis/mapping.py) for the full, authoritative mapping table.

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
