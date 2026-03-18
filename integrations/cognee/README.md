# cognee-haystack

[![PyPI](https://img.shields.io/pypi/v/cognee-haystack.svg)](https://pypi.org/project/cognee-haystack/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A [Haystack](https://haystack.deepset.ai/) integration for [Cognee](https://github.com/topoteretes/cognee) — open-source memory for AI agents.

[Cognee](https://www.cognee.ai/) gives agents persistent, traceable memory by transforming raw data (e.g., unstructured documents, relational databases, etc.) into a knowledge engine that is both searchable by meaning and connected by relationships. Cognee uses an ECL (Extract, Cognify, Load) pipeline: vector search, knowledge graphs, with support for ontology-based entity grounding (e.g. OWL) and Pydantic-defined graph schemas.

Cognee serves developers and teams building agentic systems where agents need to retain context across sessions, learn from user feedback, and operate in multi-tenant, data-isolated environments. Cognee is available as an open-source Python library (Apache 2.0), a self-hosted/on-prem subscription tier, and a managed cloud service (Cognee Cloud at platform.cognee.ai).


## Installation

```bash
pip install cognee-haystack
```

To use the `CogneeMemoryStore` with Haystack's experimental Agent:

```bash
pip install "cognee-haystack[memory]"
```

## Components

### CogneeWriter

Adds Haystack Documents to Cognee's knowledge engine via `cognee.add()`, with optional automatic `cognee.cognify()`.

```python
from haystack import Document, Pipeline
from haystack_integrations.components.connectors.cognee import CogneeWriter

pipeline = Pipeline()
pipeline.add_component("writer", CogneeWriter(dataset_name="my_data", auto_cognify=True))

docs = [Document(content="Cognee builds a structured memory from unstructured data.")]
pipeline.run({"writer": {"documents": docs}})
```

### CogneeCognifier

Standalone `cognee.cognify()` step — useful when you want to separate adding and processing.

```python
from haystack_integrations.components.connectors.cognee import CogneeCognifier

cognifier = CogneeCognifier()
cognifier.run()  # {"cognified": True}
```

### CogneeRetriever

Searches Cognee's memory and returns Haystack `Document` objects.

```python
from haystack import Pipeline
from haystack_integrations.components.connectors.cognee import CogneeRetriever

pipeline = Pipeline()
pipeline.add_component("retriever", CogneeRetriever(search_type="GRAPH_COMPLETION", top_k=5))

result = pipeline.run({"retriever": {"query": "What is Cognee?"}})
for doc in result["retriever"]["documents"]:
    print(doc.content)
```

**Supported search types:** `GRAPH_COMPLETION`, `CHUNKS`, `SUMMARIES`, `INSIGHTS`, and others from Cognee's `SearchType` enum.

### CogneeMemoryStore

Memory backend for Haystack's experimental Agent, backed by Cognee.

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.cognee import CogneeMemoryStore

store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", top_k=5)

# Store memories
store.add_memories(messages=[
    ChatMessage.from_user("The project deadline is next Friday.")
])

# Recall memories
results = store.search_memories(query="When is the deadline?")
```

## Configuration

Cognee uses environment variables for its LLM and storage configuration:

```bash
export LLM_API_KEY="sk-..."
```

See the [Cognee documentation](https://docs.cognee.ai/) for additional configuration options.

## Examples

See the [`examples/`](examples/) directory for runnable demos:

- **`demo_pipeline.py`** — Index documents and search with CogneeWriter + CogneeRetriever
- **`demo_memory_agent.py`** — Use CogneeMemoryStore as a conversational memory backend

## Development

```bash
cd integrations/cognee

# Format and lint
hatch run fmt
hatch run fmt-check

# Type checking
hatch run test:types

# Run tests
hatch run test:unit
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
