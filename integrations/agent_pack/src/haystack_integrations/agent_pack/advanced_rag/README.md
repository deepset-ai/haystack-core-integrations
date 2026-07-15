# Advanced RAG Agent

A **metadata-aware RAG agent**: instead of guessing which metadata fields exist, it first inspects
the document store (fields, values, ranges), then constructs a Haystack filter and retrieves with
it. Built on Haystack v3.

## Setup

```bash
pip install "haystack-agent-pack @ git+https://github.com/deepset-ai/haystack-agent-pack.git"
```

Set `OPENAI_API_KEY` in the environment.

## Running the agent

A fully working example — index a small corpus with varied metadata and ask a question that the
agent can only answer well by inspecting the metadata, building a filter, and retrieving with it:

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent

document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="CRISPR gene editing corrected a hereditary blindness mutation in a clinical trial.",
             meta={"category": "science", "year": 2021, "rating": 4.6, "language": "en"}),
    Document(content="A quantum computer demonstrated error-corrected logical qubits.",
             meta={"category": "science", "year": 2023, "rating": 4.8, "language": "en"}),
    Document(content="Dolly the sheep became the first mammal cloned from an adult somatic cell.",
             meta={"category": "science", "year": 1996, "rating": 4.2, "language": "en"}),
    Document(content="The Berlin Wall fell, a decisive moment in the end of the Cold War.",
             meta={"category": "history", "year": 1989, "rating": 4.7, "language": "en"}),
    Document(content="Argentina won the FIFA World Cup final against France on penalties.",
             meta={"category": "sports", "year": 2022, "rating": 4.9, "language": "en"}),
])

agent = create_advanced_rag_agent(
    document_store=document_store,
    retriever=InMemoryBM25Retriever(document_store=document_store, top_k=5),
)

result = agent.run(messages=[ChatMessage.from_user("What science advances happened after 2015?")])

print(result["last_message"].text)               # the answer, citing documents as [doc <short-id>]
for doc in result["documents"]:                  # every document the agent retrieved, deduplicated
    print(f"[doc {doc.id[:8]}] {doc.meta} :: {doc.content[:60]}")
```

The agent will list the metadata fields, verify the `category` values and the `year` range, build
a filter like `{"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==",
"value": "science"}, {"field": "meta.year", "operator": ">", "value": 2015}]}`, retrieve with it,
and answer citing the CRISPR and quantum documents.

The retrieval you provide should be **scoring-based** — keyword (BM25), embedding, or hybrid —
i.e. it ranks documents by relevance to the query. Direct, unscored fetching by metadata is
already covered by the built-in `fetch_documents_by_filter` tool.

### Using a retrieval pipeline instead of a single retriever

For anything beyond a single retriever, pass a retrieval `Pipeline` plus an input mapping that
tells the tool which sockets receive the query and the filters. A hybrid setup — BM25 + embedding
retrieval fused with reciprocal rank fusion — is what's often used in production:

```python
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever

pipeline = Pipeline()
pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=document_store))
pipeline.add_component("text_embedder", OpenAITextEmbedder())
pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=document_store))
pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
pipeline.connect("bm25_retriever.documents", "joiner.documents")
pipeline.connect("embedding_retriever.documents", "joiner.documents")

agent = create_advanced_rag_agent(
    document_store=document_store,
    retrieval_pipeline=pipeline,
    retrieval_pipeline_input_mapping={
        "query": ["bm25_retriever.query", "text_embedder.text"],
        "filters": ["bm25_retriever.filters", "embedding_retriever.filters"],
    },
    retrieval_pipeline_output_mapping={"joiner.documents": "documents"},
)
```

## Configuration

Everything is configured through keyword arguments to
[`create_advanced_rag_agent`](agent.py) — see its docstring for the full list, defaults, and
per-argument notes.

## How it works (high level)

The whole thing is a single Haystack `Agent` that loops over five tools:

```
  user question
      │
      ▼
┌──────────────────┐
│ 1. INSPECT       │  list_metadata_fields → which fields exist and their types
│    METADATA      │  get_metadata_field_values / get_metadata_field_range → what they contain
└──────┬───────────┘
       │ field names, values, ranges
       ▼
┌──────────────────┐
│ 2. FILTER +      │  search_documents(query, filters) — the filter grammar is embedded in the
│    RETRIEVE      │  tool's `filters` parameter description, so the agent builds valid filters;
│                  │  fetch_documents_by_filter(filters) — direct fetch when the exact documents
│                  │  are identifiable by metadata alone
└──────┬───────────┘
       │ documents (id + metadata + content)
       ▼
┌──────────────────┐
│ 3. ANSWER        │  answer from the retrieved documents only, citing them as [doc <short-id>]
└──────────────────┘
```

Every document any retrieval call returns is also accumulated into the agent's `State` under the
`documents` key (deduplicated by id, in first-retrieved order), so `agent.run(...)` returns the
full `list[Document]` alongside the answer. The answer cites documents by the first 8 characters
of their id (e.g. `[doc a1b2c3d4]`) — order-independent references you can resolve against the
returned list with `doc.id.startswith(...)`.

One safety net brackets the loop: if the run is cut off by `max_agent_steps` before an answer is
written, a `BackupAnswerHook` (an `after_run` hook — the only hook point that fires on step
exhaustion) makes one extra LLM call to produce a best-effort answer from the evidence gathered
so far, so `last_message` always carries a text answer.

**Its tools:**

| Tool | What it is | What it does |
|---|---|---|
| `list_metadata_fields` | `ListMetadataFieldsTool` (a `Tool` subclass over the document store) | Lists all metadata fields and their types. The system prompt instructs the agent to call this first. |
| `get_metadata_field_values` | `GetMetadataFieldValuesTool` | Returns the distinct values of a field, so filters use values that actually exist (capped listing for high-cardinality fields). |
| `get_metadata_field_range` | `GetMetadataFieldRangeTool` | Returns min/max of a numeric or orderable field (e.g. years, ratings, ISO dates). |
| `fetch_documents_by_filter` | `FetchDocumentsByFilterTool` | Fetches documents directly by metadata filter, without relevance scoring — for grabbing specific documents (e.g. a known title or file). Results are put into reading order — grouped by parent file (`file_name`/`file_path`/`source_id`) and sorted by position (`split_id`/`split_idx_start`/`page_number`) via `MetaFieldGroupingRanker` — and capped per fetch (a filter fetch has no retriever `top_k` bounding it): the LLM can pick how many documents it wants via the optional `max_docs` input, up to the configured ceiling (factory param `max_fetched_docs`). On stores that support the cheap `count_documents_by_filter` aggregation, over-broad filters (> 10 × `max_docs` matches) raise *before* fetching — surfaced to the LLM as an error tool message it can recover from — so a real database never ships thousands of documents only to show ten. |
| `search_documents` | `ComponentTool` over your retriever, or `PipelineTool` over your retrieval pipeline | Retrieves documents for a query by relevance, optionally narrowed by a metadata filter. Bounded by the `top_k` configured on your retrieval components. Returns id + metadata + content per document; an empty result nudges the agent to relax the filter. |

The four document-store-backed tools are exported individually
(`from haystack_integrations.agent_pack.advanced_rag import ListMetadataFieldsTool, FetchDocumentsByFilterTool, ...`)
and also bundled as a single unit, `DocumentStoreToolset`, so you can drop them into your own
`Agent` with your own prompt:

```python
from haystack_integrations.agent_pack.advanced_rag import DocumentStoreToolset

agent = Agent(
    chat_generator=...,
    tools=[DocumentStoreToolset(document_store), my_retrieval_tool],
)
```

## The filter grammar

The Haystack filter syntax is not something an LLM knows reliably without guidance. Rather than a
long system prompt, the grammar is embedded in the **description of the `filters` parameter** of
`search_documents`, so the model receives it contextually at the point of tool use:

- single condition: `{"field": "meta.category", "operator": "==", "value": "science"}`
- comparison operators: `==, !=, >, >=, <, <=, in, not in`
- logical grouping: `{"operator": "AND"|"OR"|"NOT", "conditions": [...]}` (nestable)
- field names must be prefixed with `meta.`

The system prompt adds the workflow rules: inspect fields first, verify values before filtering,
and relax the filter when a search comes back empty.

## Which document stores work

The metadata tools rely on optional document store methods that are not part of the base
`DocumentStore` protocol: `get_metadata_fields_info`, `get_metadata_field_unique_values`, and
`get_metadata_field_min_max`. `InMemoryDocumentStore` implements them today; integrations
(OpenSearch, etc.) will work as they adopt these methods. The tools fail fast at construction time
with a clear error if the store doesn't support them.

## Serialization note

The agent and every tool support `to_dict`/`from_dict`, and deserialize out of the box: the
`haystack_integrations` namespace is on Haystack's default deserialization allowlist.

## Possible follow-up

- **Auto-guess retrieval mode**: call the factory with only a `document_store` and let it infer a
  retrieval setup (e.g. `InMemoryDocumentStore` → `InMemoryBM25Retriever`). For now the factory
  requires exactly one of `retriever` / `retrieval_pipeline` to stay predictable.
