# Cognee-Haystack Examples

## Prerequisites

Install the integration from the repository root:

```bash
pip install -e "integrations/cognee"
```

Set your LLM API key (required by cognee, default OpenAI API key):

To integrate other LLM providers and other configuration options, see [Cognee Documentation](https://docs.cognee.ai/getting-started/installation#environment-configuration).


```bash
export LLM_API_KEY="sk-your-openai-api-key"
```

## Examples

### Pipeline Demo (`demo_pipeline.py`)

Demonstrates batch document ingestion with `CogneeWriter` (auto_cognify disabled),
followed by a single `CogneeCognifier` pass, then retrieval with `CogneeRetriever`.
Also shows the same flow wired as a connected Haystack Pipeline.

```bash
python integrations/cognee/examples/demo_pipeline.py
```

### Memory Agent Demo (`demo_memory_agent.py`)

Demonstrates `CogneeMemoryStore` with per-user memory scoping via `user_id`:
- Two users store private memories that are isolated from each other.
- A shared dataset is created and read access is granted across users.

```bash
python integrations/cognee/examples/demo_memory_agent.py
```
