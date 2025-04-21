import os
import logging
from pymongo import MongoClient, TEXT
from pymongo.operations import SearchIndexModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from requests.auth import HTTPDigestAuth

# Logging for visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embedding_dimension = 768


DEFAULT_DOCS = [
    {"content": "Document A", "embedding": [-1] + [0.2] * (embedding_dimension - 1)},
    {"content": "Document B", "embedding": [0] + [0.15] * (embedding_dimension - 1)},
    {"content": "Document C", "embedding": [0.1] * embedding_dimension},
]

VECTOR_INDEXES = [
    {
        "name": "cosine_index",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
            {"type": "filter", "path": "content"},
        ],
    },
    {
        "name": "dotProduct_index",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "dotProduct"},
        ],
    },
    {
        "name": "euclidean_index",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "euclidean"},
        ],
    },
]

FULL_TEXT_INDEX = {
    "name": "full_text_index",
    "definition": {
        "mappings": {"dynamic": True},
    },
}


def get_collection(client, db_name, coll_name):
    db = client[db_name]
    if coll_name not in db.list_collection_names():
        logger.info(f"Creating collection '{coll_name}' in DB '{db_name}'")
        db.create_collection(coll_name)
    return db[coll_name]


def setup_test_embeddings_collection(client):
    collection = get_collection(client, "haystack_integration_test", "test_embeddings_collection")

    if collection.count_documents({}) == 0:
        collection.insert_many(DEFAULT_DOCS)

    existing_index_names = {idx["name"] for idx in collection.list_search_indexes()}

    for index in VECTOR_INDEXES:
        if index["name"] not in existing_index_names:
            logger.info(f"Creating vector search index: {index['name']}")
            model = SearchIndexModel(definition={"fields": index["fields"]}, name=index["name"], type="vectorSearch")
            collection.create_search_index(model=model)


def setup_test_full_text_search_collection(client):
    collection = get_collection(client, "haystack_integration_test", "test_full_text_search_collection")

    existing_index_names = {idx["name"] for idx in collection.list_search_indexes()}

    if FULL_TEXT_INDEX["name"] not in existing_index_names:
        logger.info(f"Creating full text search index: {FULL_TEXT_INDEX['name']}")
        model = SearchIndexModel(definition=FULL_TEXT_INDEX["definition"], name=FULL_TEXT_INDEX["name"], type="search")
        collection.create_search_index(model=model)


def setup_mongodb_for_tests():
    connection_str = os.environ.get("MONGO_CONNECTION_STRING")
    if not connection_str:
        logger.warning("Skipping MongoDB Atlas setup: no MONGO_CONNECTION_STRING")
        return

    client = MongoClient(connection_str)

    setup_test_embeddings_collection(client)
    setup_test_full_text_search_collection(client)


if __name__ == "__main__":
    setup_mongodb_for_tests()
