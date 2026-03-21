#!/usr/bin/env python
"""
Demo: Cognee + Haystack Pipeline

Shows how to use CogneeWriter and CogneeRetriever in a Haystack pipeline
to ingest documents into Cognee's memory and search them.

Prerequisites:
    pip install -e "integrations/cognee"

Set your LLM API key (Cognee uses it internally):
    export LLM_API_KEY="sk-..."
"""

import asyncio
import os

from cognee.api.v1.visualize.visualize import visualize_graph
from haystack import Document, Pipeline

from haystack_integrations.components.retrievers.cognee import CogneeRetriever
from haystack_integrations.components.writers.cognee import CogneeWriter

SAMPLE_DOCUMENTS = [
    Document(
        content=(
            "Cognee is an open-source memory for AI agents. Cognee builts a knowledge engine"
            "that transforms raw data (e.g., unstructured documents, relational databases, etc.)"
            "into a persistent, rich, and traceable memory that is searchable by meaning and relationships."
        ),
        meta={"topic": "cognee"},
    ),
    Document(
        content=(
            "Haystack is an open-source LLM framework by deepset for building production-ready "
            "RAG pipelines, agents, and search systems. It uses a component-based architecture "
            "where each step (retrieval, generation, etc.) is a composable building block."
        ),
        meta={"topic": "haystack"},
    ),
    Document(
        content=(
            "Knowledge graphs represent information as nodes (entities) and edges (relationships). "
            "They enable semantic search, reasoning, and discovery of hidden connections across "
            "large document collections."
        ),
        meta={"topic": "knowledge_graphs"},
    ),
    Document(
        content=(
            "The engineering team at Acme Corp consists of Alice (backend lead), Bob (ML engineer), "
            "and Carol (infrastructure). They are building a next-generation search platform "
            "powered by knowledge graphs and LLMs."
        ),
        meta={"topic": "team"},
    ),
]


async def main():
    print("=== Cognee + Haystack Pipeline Demo ===\n")

    # --- Indexing pipeline ---
    print("1. Building indexing pipeline...")
    indexing = Pipeline()
    indexing.add_component("writer", CogneeWriter(dataset_name="demo", auto_cognify=True))

    print(f"2. Indexing {len(SAMPLE_DOCUMENTS)} documents...")
    result = indexing.run({"writer": {"documents": SAMPLE_DOCUMENTS}})
    print(f"   Written: {result['writer']['documents_written']} documents\n")

    visualization_path = os.path.join(os.path.dirname(__file__), ".artifacts", "demo_pipeline.html")
    await visualize_graph(visualization_path)

    # --- Query pipeline ---
    print("3. Building query pipeline...")
    querying = Pipeline()
    querying.add_component("retriever", CogneeRetriever(search_type="GRAPH_COMPLETION", top_k=5))

    queries = [
        "What is Cognee and what does it do?",
        "Who is on the engineering team at Acme Corp?",
        "How do knowledge graphs work?",
    ]

    for query in queries:
        print(f"4. Searching: '{query}'")
        result = querying.run({"retriever": {"query": query}})
        docs = result["retriever"]["documents"]
        print(f"   Found {len(docs)} result(s):")
        for i, doc in enumerate(docs, 1):
            print(f"   [{i}] {doc.content}...")
        print()

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
