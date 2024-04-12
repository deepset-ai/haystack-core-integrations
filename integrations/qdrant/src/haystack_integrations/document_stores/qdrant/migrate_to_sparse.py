import time

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.http import models

# old_collection_name = "Document"
new_collection_name = "Document_sparse"

# old_document_store = QdrantDocumentStore(
#     url="http://localhost:6333",
#     index="Document",
#     use_sparse_embeddings=False,
# )

def migrate_to_sparse_embeddings_support(old_document_store: QdrantDocumentStore, new_index: str):

    start = time.time()

    old_collection_name = old_document_store.index

    total_points = old_document_store.count_documents()

    
    # copy the init parameters of the old document to create a new document store
    init_parameters = old_document_store.to_dict()["init_parameters"]
    init_parameters["index"] = new_collection_name
    init_parameters["use_sparse_embeddings"] = True
    init_parameters["recreate_index"] = True


    new_document_store = QdrantDocumentStore(**init_parameters)

    client = new_document_store.client

    original_optimizer_config = client.get_collection(collection_name=new_collection_name).config.optimizer_config

    # Disable indexing while adding points so it's faster
    # https://qdrant.tech/documentation/concepts/collections/#update-collection-parameters
    client.update_collection(
        collection_name=new_collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    next_page_offset = "first"
    offset = None
    points_transmitted = 0

    while next_page_offset:
        if next_page_offset != "first":
            offset = next_page_offset

        # get the records
        records = client.scroll(
            collection_name=old_collection_name,
            limit=100,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )

        next_page_offset = records[1]
        current_records = records[0]

        points = []

        for record in current_records:
            vector = {}

            vector["text-dense"] = record.vector

            point = {"id": record.id, "payload": record.payload, "vector": vector}

            embedding_point = models.PointStruct(**point)
            points.append(embedding_point)

        client.upsert(collection_name=new_collection_name, points=points)

        points_transmitted += len(points)
        points_remaining = total_points - points_transmitted

        # Print progress
        message = f"Points transmitted: {points_transmitted}/{total_points}\n"
                  f"Percent done {points_transmitted/total_points*100:.2f}%\n"
                  f"Time elapsed: {time.time() - start:.2f} seconds\n"
                  f"Time remaining: {(((time.time() - start) / points_transmitted) * points_remaining) / 60:.2f} minutes\n"
                  f"Current offset: {next_page_offset}"
        print(message)

    # restore the original optimizer config (re-enable indexing)
    client.update_collection(
        collection_name=new_collection_name,
        optimizer_config=original_optimizer_config,
    )
