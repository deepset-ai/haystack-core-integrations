# Advanced RAG Agent

A **metadata-aware RAG agent**: instead of guessing which metadata fields exist, it inspects the
document store (fields, values, ranges) and can construct Haystack filters to narrow its
retrieval. Built on Haystack v3.

## Setup

```bash
pip install agent-pack-haystack
```

Set `OPENAI_API_KEY` in the environment.

## Running the agent

Index a corpus with varied metadata and ask a question the agent can only answer well by
inspecting the metadata, building a filter, and retrieving with it:

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.agent_pack import create_advanced_rag_agent

document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="CRISPR gene editing corrected a hereditary blindness mutation in a clinical trial.",
             meta={"category": "science", "year": 2021, "rating": 4.6}),
    Document(content="A quantum computer demonstrated error-corrected logical qubits.",
             meta={"category": "science", "year": 2023, "rating": 4.8}),
    Document(content="Dolly the sheep became the first mammal cloned from an adult somatic cell.",
             meta={"category": "science", "year": 1996, "rating": 4.2}),
    Document(content="The Berlin Wall fell, a decisive moment in the end of the Cold War.",
             meta={"category": "history", "year": 1989, "rating": 4.7}),
    Document(content="Argentina won the FIFA World Cup final against France on penalties.",
             meta={"category": "sports", "year": 2022, "rating": 4.9}),
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

The agent lists the metadata fields, verifies the `category` values and the `year` range, builds a
filter like `{"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==",
"value": "science"}, {"field": "meta.year", "operator": ">", "value": 2015}]}`, retrieves with it,
and answers citing the CRISPR and quantum documents. Filtering is optional — when metadata can't
narrow a question, the agent retrieves without one.

The retrieval you provide should be **scoring-based** — keyword (BM25), embedding, or hybrid.
Direct, unscored fetching by metadata is already covered by the built-in
`fetch_documents_by_filter` tool.

### Using a retrieval pipeline instead of a single retriever

For anything beyond a single retriever, pass a retrieval `Pipeline` plus an input mapping that
tells the tool which sockets receive the query and the filters — e.g. hybrid retrieval with
reciprocal rank fusion:

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
│ 2. RETRIEVE      │  search_documents(query, filters) — relevance search, optionally filtered;
│                  │  fetch_documents_by_filter(filters) — direct fetch when the exact documents
│                  │  are identifiable by metadata alone
└──────┬───────────┘
       │ documents (id + metadata + content)
       ▼
┌──────────────────┐
│ 3. ANSWER        │  answer from the retrieved documents only, citing them as [doc <short-id>]
└──────────────────┘
```

Every retrieved document is accumulated into the agent's `State` under the `documents` key
(deduplicated by id), so `agent.run(...)` returns the full `list[Document]` alongside the answer.
The answer cites documents by the first 8 characters of their id (e.g. `[doc a1b2c3d4]`) —
resolve them against the returned list with `doc.id.startswith(...)`.

If the run is cut off by `max_agent_steps` before an answer is written, a `BackupAnswerHook`
(an `after_run` hook) makes one extra LLM call to produce a best-effort answer from the evidence
gathered so far, so `last_message` always carries a text answer.

**Its tools:**

| Tool | What it is | What it does |
|---|---|---|
| `list_metadata_fields` | `ListMetadataFieldsTool` | Lists all metadata fields and their types. The system prompt instructs the agent to call this first. |
| `get_metadata_field_values` | `GetMetadataFieldValuesTool` | Returns the distinct values of a field, so filters use values that actually exist. The listing is capped; an optional `search_term` narrows the values by case-insensitive substring (applied client-side, so the semantics are identical on every store). |
| `get_metadata_field_range` | `GetMetadataFieldRangeTool` | Returns min/max of a numeric or orderable field (e.g. years, ratings, ISO dates). |
| `fetch_documents_by_filter` | `FetchDocumentsByFilterTool` | Fetches documents directly by metadata filter, without relevance scoring — for grabbing specific documents (e.g. a known title or file). Results are put into reading order (grouped by parent file, sorted by split/page) and capped at `max_docs` per fetch; over-broad filters are refused *before* fetching, as an error the LLM recovers from by narrowing. |
| `search_documents` | `ComponentTool` over your retriever, or `PipelineTool` over your retrieval pipeline | Retrieves documents for a query by relevance, optionally narrowed by a metadata filter. Bounded by the `top_k` of your retrieval components; an empty result nudges the agent to relax the filter. |

The four document-store-backed tools are exported individually and also bundled as
`DocumentStoreToolset`, so you can drop them into your own `Agent` with your own prompt:

```python
from haystack_integrations.agent_pack.advanced_rag import DocumentStoreToolset

agent = Agent(
    chat_generator=...,
    tools=[DocumentStoreToolset(document_store), my_retrieval_tool],
)
```

The agent and every tool serialize via `to_dict`/`from_dict`.

## The filter grammar

The Haystack filter syntax is not something an LLM knows reliably without guidance. Rather than a
long system prompt, the grammar is embedded in the **description of the `filters` parameter** of
`search_documents` and `fetch_documents_by_filter`, so the model receives it contextually at the
point of tool use:

- single condition: `{"field": "meta.category", "operator": "==", "value": "science"}`
- comparison operators: `==, !=, >, >=, <, <=, in, not in`
- logical grouping: `{"operator": "AND"|"OR"|"NOT", "conditions": [...]}` (nestable)
- field names must be prefixed with `meta.`

The system prompt adds the workflow rules: inspect fields first, verify values before filtering,
and relax the filter when a search comes back empty.

## Which document stores work

The metadata tools rely on document store methods that are not part of the base `DocumentStore`
protocol: `get_metadata_fields_info`, `get_metadata_field_unique_values`, and
`get_metadata_field_min_max`. `InMemoryDocumentStore` and most document store integrations
implement them (OpenSearch, Elasticsearch, Weaviate, Chroma, pgvector, Qdrant, Pinecone,
MongoDB Atlas, Astra, and more). Each tool fails fast at construction time with a clear error if
the store doesn't support the method it needs — stores that implement only some of the methods can
still use the matching subset of tools.
