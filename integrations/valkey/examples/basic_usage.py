# Basic usage example for Valkey Document Store
# This example shows how to use ValkeyDocumentStore for basic document operations

from haystack.dataclasses import Document

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

# Initialize ValkeyDocumentStore with basic configuration
document_store = ValkeyDocumentStore(
    nodes_list=[("localhost", 6379)],
    index_name="basic_example",
    embedding_dim=384,  # Smaller dimension for this example
    distance_metric="cosine",
)

# Create some sample documents with embeddings and metadata
documents = [
    Document(
        id="doc_1",
        content="Artificial Intelligence is transforming the world.",
        embedding=[0.1, 0.2, 0.3] + [0.0] * 381,  # 384-dimensional vector
        meta={"category": "technology", "priority": 1, "score": 0.9},
    ),
    Document(
        id="doc_2",
        content="Machine Learning is a subset of AI.",
        embedding=[0.2, 0.3, 0.4] + [0.0] * 381,
        meta={"category": "technology", "priority": 2, "score": 0.8},
    ),
    Document(
        id="doc_3",
        content="Deep Learning uses neural networks.",
        embedding=[0.3, 0.4, 0.5] + [0.0] * 381,
        meta={"category": "research", "priority": 1, "score": 0.95},
    ),
    Document(
        id="doc_4",
        content="Natural Language Processing enables computers to understand text.",
        embedding=[0.4, 0.5, 0.6] + [0.0] * 381,
        meta={"category": "nlp", "priority": 3, "score": 0.7},
    ),
]

# Write documents to the store
print("Writing documents...")
written_count = document_store.write_documents(documents)
print(f"Successfully wrote {written_count} documents")

# Count total documents
total_docs = document_store.count_documents()
print(f"Total documents in store: {total_docs}")

# Search by embedding similarity
print("\n--- Embedding Search ---")
query_embedding = [0.15, 0.25, 0.35] + [0.0] * 381
search_results = document_store.search(query_embedding, limit=2)

for i, doc in enumerate(search_results, 1):
    print(f"{i}. {doc.content}")
    print(f"   Score: {doc.score:.4f}")
    print(f"   Category: {doc.meta.get('category')}")

# Filter documents by metadata
print("\n--- Metadata Filtering ---")
filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "technology"}]}

filtered_docs = document_store.filter_documents(filters)
print(f"Found {len(filtered_docs)} technology documents:")
for doc in filtered_docs:
    print(f"- {doc.content}")

# Combined search with filters
print("\n--- Search with Filters ---")
filters = {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": "<=", "value": 2}]}

filtered_search = document_store.search(query_embedding, filters=filters, limit=3)
print(f"Found {len(filtered_search)} documents with priority <= 2:")
for doc in filtered_search:
    print(f"- {doc.content} (Priority: {doc.meta.get('priority')})")

# Delete specific documents
print("\n--- Document Deletion ---")
document_store.delete_documents(["doc_2", "doc_4"])
remaining_docs = document_store.count_documents()
print(f"Documents remaining after deletion: {remaining_docs}")

# Clean up - delete all documents
document_store.delete_all_documents()
final_count = document_store.count_documents()
print(f"Documents after cleanup: {final_count}")

print("\nExample completed successfully!")
