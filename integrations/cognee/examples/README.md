# Cognee-Haystack Examples

## Prerequisites

Install the integration from the repository root:

```bash
pip install -e "integrations/cognee[memory]"
```

Set your LLM API key (required by cognee, default OpenAI API key) and ENABLE_BACKEND_ACCESS_CONTROL=False for simplicity:

To integrate other LLM providers and other configuration options, see [Cognee Documentation](https://docs.cognee.ai/getting-started/installation#environment-configuration).


```bash
export LLM_API_KEY="sk-your-openai-api-key"
export ENABLE_BACKEND_ACCESS_CONTROL="False"
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
