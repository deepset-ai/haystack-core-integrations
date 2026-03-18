# Cognee-Haystack Examples

## Prerequisites

Install the integration from the repository root:

```bash
pip install -e "integrations/cognee[memory]"
```

Set your LLM API key (required by cognee):

```bash
export LLM_API_KEY="sk-..."
```

## Examples

### Pipeline Demo (`demo_pipeline.py`)

Demonstrates `CogneeWriter` and `CogneeRetriever` in a Haystack pipeline:

```bash
python integrations/cognee/examples/demo_pipeline.py
```

### Memory Agent Demo (`demo_memory_agent.py`)

Demonstrates `CogneeMemoryStore` as a conversational memory backend:

```bash
python integrations/cognee/examples/demo_memory_agent.py
```
