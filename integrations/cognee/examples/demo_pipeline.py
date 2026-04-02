#!/usr/bin/env python
"""
Demo: Cognee + Haystack Pipeline — Writer, Cognifier, Retriever

Demonstrates two ingestion strategies with CogneeWriter:

1. auto_cognify=True (default) — each writer.run() call adds documents AND
   immediately cognifies them into the knowledge graph. Simple but slower
   when ingesting many batches, because cognify runs after every add.

2. auto_cognify=False + CogneeCognifier — documents are added quickly in
   multiple batches without building the knowledge graph. Then CogneeCognifier
   processes the entire dataset once. This is faster for bulk ingestion.

Both strategies produce the same result: a searchable knowledge graph that
CogneeRetriever can query.

Prerequisites:
    pip install -e "integrations/cognee"

Set your LLM API key (Cognee uses it internally):
    export LLM_API_KEY="sk-..."
"""

import asyncio

import cognee
from haystack import Document, Pipeline

from haystack_integrations.components.connectors.cognee import CogneeCognifier
from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.components.writers.cognee import CogneeWriter

DOCS_BATCH_1 = [
    Document(
        content=(
            "Cognee is an open-source memory for AI agents. It builds a knowledge engine "
            "that transforms raw data into a persistent, rich, and traceable memory that is "
            "searchable by meaning and relationships."
        ),
    ),
    Document(
        content=(
            "Haystack is an open-source LLM framework by deepset for building production-ready "
            "RAG pipelines, agents, and search systems. It uses a component-based architecture "
            "where each step is a composable building block."
        ),
    ),
]

DOCS_BATCH_2 = [
    Document(
        content=(
            "Knowledge graphs represent information as nodes (entities) and edges (relationships). "
            "They enable semantic search, reasoning, and discovery of hidden connections across "
            "large document collections."
        ),
    ),
    Document(
        content=(
            "The engineering team at Acme Corp consists of Alice (backend lead), Bob (ML engineer), "
            "and Carol (infrastructure). They are building a next-generation search platform "
            "powered by knowledge graphs and LLMs."
        ),
    ),
]

SEARCH_QUERIES = [
    "What is Cognee and what does it do?",
    "Who is on the engineering team at Acme Corp?",
    "How do knowledge graphs work?",
]


def search_and_print(retriever, queries):
    for query in queries:
        print(f"   Query: '{query}'")
        result = retriever.run(query=query)
        docs = result["documents"]
        print(f"   Found {len(docs)} result(s):")
        for i, doc in enumerate(docs, 1):
            print(f"   [{i}] {doc.content}...")
        print()


async def main():
    print("=== Cognee + Haystack Pipeline Demo ===\n")

    # =========================================================================
    # Part 1: auto_cognify=True — add + cognify on every call
    #
    # Each writer.run() both adds the documents AND cognifies the dataset.
    # Simple for small ingestion, but cognify is the expensive step (calls
    # the LLM to extract entities, build the graph, generate summaries).
    # With N batches, cognify runs N times.
    # =========================================================================
    print("--- Part 1: auto_cognify=True (add + cognify each batch) ---\n")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    writer_auto = CogneeWriter(dataset_name="demo_auto", auto_cognify=True)

    print(f"1. Writing batch 1 ({len(DOCS_BATCH_1)} docs) — adds + cognifies...")
    result = writer_auto.run(documents=DOCS_BATCH_1)
    print(f"   Written: {result['documents_written']} (cognify ran)\n")

    print(f"2. Writing batch 2 ({len(DOCS_BATCH_2)} docs) — adds + cognifies again...")
    result = writer_auto.run(documents=DOCS_BATCH_2)
    print(f"   Written: {result['documents_written']} (cognify ran again)\n")

    print("3. Searching...\n")
    retriever = CogneeRetriever(search_type="GRAPH_COMPLETION", dataset_name="demo_auto")
    search_and_print(retriever, SEARCH_QUERIES)

    # =========================================================================
    # Part 2: auto_cognify=False + CogneeCognifier — batch add, cognify once
    #
    # Documents are added quickly without cognifying. Then CogneeCognifier
    # processes everything in one pass. With N batches, cognify runs only once.
    # =========================================================================
    print("--- Part 2: auto_cognify=False + CogneeCognifier (batch add, cognify once) ---\n")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    writer_batch = CogneeWriter(dataset_name="demo_batch", auto_cognify=False)

    print(f"4. Writing batch 1 ({len(DOCS_BATCH_1)} docs) — add only, no cognify...")
    result = writer_batch.run(documents=DOCS_BATCH_1)
    print(f"   Written: {result['documents_written']}\n")

    print(f"5. Writing batch 2 ({len(DOCS_BATCH_2)} docs) — add only, no cognify...")
    result = writer_batch.run(documents=DOCS_BATCH_2)
    print(f"   Written: {result['documents_written']}\n")

    print("6. Cognifying the entire dataset in one pass...")
    cognifier = CogneeCognifier(dataset_name="demo_batch")
    result = cognifier.run()
    print(f"   Cognified: {result['cognified']}\n")

    print("7. Searching...\n")
    retriever = CogneeRetriever(search_type="GRAPH_COMPLETION", dataset_name="demo_batch")
    search_and_print(retriever, SEARCH_QUERIES)

    # =========================================================================
    # Part 3: Same batch flow wired as a Haystack Pipeline
    #
    # CogneeWriter(auto_cognify=False) outputs documents_written, which
    # connects to CogneeCognifier's input — so cognify triggers automatically
    # after the writer finishes.
    # =========================================================================
    print("--- Part 3: Writer + Cognifier as a connected Pipeline ---\n")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    pipeline = Pipeline()
    pipeline.add_component("writer", CogneeWriter(dataset_name="demo_pipeline", auto_cognify=False))
    pipeline.add_component("cognifier", CogneeCognifier(dataset_name="demo_pipeline"))
    pipeline.connect("writer.documents_written", "cognifier.documents_written")

    all_docs = DOCS_BATCH_1 + DOCS_BATCH_2
    print(f"8. Running pipeline with {len(all_docs)} documents...")
    result = pipeline.run({"writer": {"documents": all_docs}})
    print(f"   Cognified: {result['cognifier']['cognified']}\n")

    print("9. Searching...\n")
    retriever = CogneeRetriever(search_type="GRAPH_COMPLETION", dataset_name="demo_pipeline")
    search_and_print(retriever, SEARCH_QUERIES)

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
