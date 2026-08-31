# Examples

Three runnable scripts, smallest first. Each traces a different Haystack shape, so the span trees
they produce look quite different from one another.

| Script | Shape | What the trace looks like |
|---|---|---|
| [`chat.py`](chat.py) | A minimal chat pipeline | The smallest complete setup: a `function.haystack.pipeline.run` root with an `ai.llm.invoke` child. Shows passing a `session_id` through the connector's `invocation_context` input, and the `trace_url` / `trace_id` it returns. |
| [`basic_rag.py`](basic_rag.py) | A retrieval pipeline | Adds `ai.retrieval` alongside `ai.llm.invoke` under the pipeline root, so you can see how component types map onto Rhesis span names. |
| [`agent.py`](agent.py) | A standalone `Agent`, no pipeline | The agent loop: an `ai.agent.invoke` root, one `ai.llm.invoke` per step, and one `ai.tool.invoke` per tool call. Because there is no pipeline to carry an `invocation_context` socket, metadata is attached with `rhesis_invocation_context` instead. |

## Setup

You need a Rhesis account and an API key. Sign up at [rhesis.ai](https://rhesis.ai), or point
`RHESIS_BASE_URL` at a self-hosted backend. Docs: [docs.rhesis.ai](https://docs.rhesis.ai).

```bash
pip install -r requirements.txt

export RHESIS_API_KEY=your-rhesis-key
export OPENAI_API_KEY=your-openai-key
```

Optional, for a self-hosted or local backend:

```bash
export RHESIS_BASE_URL=http://localhost:8080
export RHESIS_FRONTEND_URL=http://localhost:3000   # makes `trace_url` resolvable
```

Then run any of them:

```bash
python chat.py
```

## Finding the trace

`chat.py` and `basic_rag.py` print the `trace_url` the connector returns — open it to land directly
on the trace. `agent.py` prints only the agent's reply, since a standalone `Agent` has no connector
output socket; find its trace in the Rhesis UI under the name passed to `RhesisConnector`, in this
case `Agent example`.

## A note on content

All three scripts set `HAYSTACK_CONTENT_TRACING_ENABLED=true` before they import `haystack`. That
ordering matters: Haystack reads the variable once, when `haystack.tracing` is first
imported, so setting it afterwards has no effect and prompts and completions would not appear on the
spans. It also means these examples send prompt and reply text to Rhesis as written.
