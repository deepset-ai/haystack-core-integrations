# Cognee-Haystack Examples

## Prerequisites

Install the integration from the repository root:

```bash
pip install -e "integrations/cognee"
```

Set the keys used by cognee and by Haystack's `OpenAIChatGenerator`:

```bash
export LLM_API_KEY="sk-your-openai-api-key"        # cognee: remember/recall/improve
export EMBEDDING_API_KEY="sk-your-openai-api-key"  # cognee: embeddings (often same key)
export OPENAI_API_KEY="sk-your-openai-api-key"     # Haystack OpenAIChatGenerator (Agent)
```

In practice all three can point at the same OpenAI key. For other LLM providers and full
configuration options, see [Cognee Documentation](https://docs.cognee.ai/getting-started/installation#environment-configuration).

cognee's session cache is on by default (`CACHING=true`, `CACHE_BACKEND=fs`); set
`CACHE_BACKEND=redis` plus `CACHE_HOST` / `CACHE_PORT` / `CACHE_USERNAME` /
`CACHE_PASSWORD` to point at a Redis instance instead.

## Examples

### Memory Agent Demo (`demo_memory_agent.py`)

Wires `CogneeRetriever`, `CogneeMemoryStore`, and `CogneeWriter` around Haystack's
`Agent` to enrich every conversation turn with persistent memory, in four phases:

1. `persistent_writer` seeds long-lived facts into the permanent graph.
2. `session_writer` seeds session-only context (`session_id=...`).
3. Agent loop: `CogneeRetriever` calls `cognee.recall(query, session_id=...)` which
   auto-captures each turn as a QA entry in the session — no writer in the pipeline.
4. `chat_store.improve()` promotes the session into the permanent graph.

```bash
python integrations/cognee/examples/demo_memory_agent.py
```
