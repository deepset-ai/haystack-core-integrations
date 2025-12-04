# Async operations example for Valkey Document Store
# This example demonstrates asynchronous document operations

import asyncio

from haystack.dataclasses import Document

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


async def main():
    # Initialize ValkeyDocumentStore
    document_store = ValkeyDocumentStore(
        nodes_list=[("localhost", 6379)],
        index_name="async_example",
        embedding_dim=256,
        distance_metric="cosine",
        batch_size=10,  # Smaller batch size for demo
    )

    # Create sample documents
    documents = []
    for i in range(25):  # Create 25 documents to test batching
        documents.append(
            Document(
                id=f"async_doc_{i}",
                content=f"This is async document number {i} about various topics.",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i] + [0.0] * 253,
                meta={"category": "async" if i % 2 == 0 else "batch", "priority": i % 5, "score": 0.5 + (i * 0.02)},
            )
        )

    try:
        # Async write operations (will be batched automatically)
        print("Writing documents asynchronously...")
        written_count = await document_store.async_write_documents(documents)
        print(f"Successfully wrote {written_count} documents")

        # Async count documents
        total_docs = await document_store.async_count_documents()
        print(f"Total documents in store: {total_docs}")

        # Async search by embedding
        print("\n--- Async Embedding Search ---")
        query_embedding = [0.5, 1.0, 1.5] + [0.0] * 253
        search_results = await document_store.async_search(query_embedding, limit=3)

        for i, doc in enumerate(search_results, 1):
            print(f"{i}. {doc.content}")
            print(f"   Score: {doc.score:.4f}")

        # Async filter documents
        print("\n--- Async Metadata Filtering ---")
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "async"},
                {"field": "meta.priority", "operator": ">=", "value": 2},
            ],
        }

        filtered_docs = await document_store.async_filter_documents(filters)
        print(f"Found {len(filtered_docs)} async documents with priority >= 2:")
        for doc in filtered_docs[:3]:  # Show first 3
            print(f"- {doc.content} (Priority: {doc.meta.get('priority')})")

        # Async delete specific documents
        print("\n--- Async Document Deletion ---")
        docs_to_delete = [f"async_doc_{i}" for i in range(0, 5)]
        await document_store.async_delete_documents(docs_to_delete)

        remaining_docs = await document_store.async_count_documents()
        print(f"Documents remaining after deletion: {remaining_docs}")

        # Async search with filters
        print("\n--- Async Search with Filters ---")
        priority_filter = {
            "operator": "AND",
            "conditions": [{"field": "meta.priority", "operator": "in", "value": [1, 3]}],
        }

        filtered_search = await document_store.async_search(query_embedding, filters=priority_filter, limit=5)
        print(f"Found {len(filtered_search)} documents with priority 1 or 3:")
        for doc in filtered_search:
            print(f"- ID: {doc.id}, Priority: {doc.meta.get('priority')}")

    finally:
        # Clean up - delete all documents
        print("\n--- Cleanup ---")
        await document_store.async_delete_all_documents()
        final_count = await document_store.async_count_documents()
        print(f"Documents after cleanup: {final_count}")

        # Close the async connection
        await document_store.async_close()
        print("Async connection closed")

    print("\nAsync example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
