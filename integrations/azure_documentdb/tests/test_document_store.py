# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest
from azure.core.credentials import AccessToken
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseExtendedTests
from haystack.utils import Secret
from pymongo import InsertOne, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError

from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore
from haystack_integrations.document_stores.azure_documentdb.document_store import AzureIdentityTokenCallback


def test_default_authentication_uses_entra_id(monkeypatch):
    monkeypatch.setenv("AZURE_DOCUMENTDB_CLUSTER_NAME", "cluster")
    credential = Mock()
    store = AzureDocumentDBDocumentStore(
        database_name="db", collection_name="collection", azure_token_credential=credential
    )
    uri, kwargs = store._client_kwargs()
    assert uri == "mongodb+srv://cluster.global.mongocluster.cosmos.azure.com/"
    assert kwargs["authMechanism"] == "MONGODB-OIDC"
    assert kwargs["retryWrites"] is False
    assert kwargs["authMechanismProperties"]["OIDC_CALLBACK"]._credential is credential


def test_missing_cluster_name_raises(monkeypatch):
    monkeypatch.delenv("AZURE_DOCUMENTDB_CLUSTER_NAME", raising=False)
    store = AzureDocumentDBDocumentStore(database_name="db", collection_name="collection")
    with pytest.raises(DocumentStoreError, match="cluster name is required"):
        store._client_kwargs()


def test_connection_string_fallback_warns(caplog):
    store = AzureDocumentDBDocumentStore(
        database_name="db",
        collection_name="collection",
        mongo_connection_string=Secret.from_token("mongodb://localhost"),
    )
    uri, kwargs = store._client_kwargs()
    assert uri == "mongodb://localhost"
    assert kwargs["retryWrites"] is False
    assert "intended only for local development" in caplog.text


def test_oidc_callback_fetches_documentdb_scope():
    credential = Mock()
    credential.get_token.return_value = AccessToken("token", 1234)
    result = AzureIdentityTokenCallback(credential).fetch(Mock())
    credential.get_token.assert_called_once_with("https://ossrdbms-aad.database.windows.net/.default")
    assert result.access_token == "token"


def test_serialization():
    store = AzureDocumentDBDocumentStore(
        database_name="db", collection_name="collection", cluster_name="cluster", mongo_connection_string=None
    )
    serialized = store.to_dict()
    restored = AzureDocumentDBDocumentStore.from_dict(serialized)
    assert restored.database_name == "db"
    assert restored.collection_name == "collection"
    assert restored.cluster_name == "cluster"
    assert restored.mongo_connection_string is None


def test_embedding_pipeline_uses_cosmos_search(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    collection.aggregate.return_value = [
        {"document": {"_id": "internal", "id": "doc", "content": "text", "meta": {}}, "score": 0.9}
    ]
    documents = store._embedding_retrieval(
        [0.1, 0.2], filters={"field": "meta.kind", "operator": "==", "value": "guide"}, top_k=3
    )
    pipeline = collection.aggregate.call_args.args[0]
    assert pipeline[0] == {
        "$search": {
            "cosmosSearch": {
                "vector": [0.1, 0.2],
                "path": "embedding",
                "k": 3,
                "filter": {"meta.kind": {"$eq": "guide"}},
            },
            "returnStoredSource": True,
        }
    }
    assert documents == [Document(id="doc", content="text", meta={}, score=0.9)]


def test_full_text_pipeline_follows_documentdb_order(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    store._full_text_retrieval("azure", fuzzy={"maxEdits": 1}, top_k=2)
    pipeline = collection.aggregate.call_args.args[0]
    assert pipeline[0] == {
        "$search": {
            "index": "text_index",
            "text": {"query": "azure", "path": "content", "fuzzy": {"maxEdits": 1}},
        }
    }
    assert pipeline[1] == {"$limit": 2}


def test_create_vector_index(mocked_store_collection):
    store, _, _, database = mocked_store_collection
    store.create_vector_index(dimensions=1536, kind="vector-diskann", similarity="COS", maxDegree=32, lBuild=50)
    database.command.assert_called_once_with(
        {
            "createIndexes": "test_collection",
            "indexes": [
                {
                    "name": "vector_index",
                    "key": {"embedding": "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": "vector-diskann",
                        "dimensions": 1536,
                        "similarity": "COS",
                        "maxDegree": 32,
                        "lBuild": 50,
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("similarity", ["COS", "L2", "IP"])
def test_create_vector_index_supports_documentdb_similarities(mocked_store_collection, similarity):
    store, _, _, database = mocked_store_collection
    store.create_vector_index(dimensions=3, similarity=similarity)
    options = database.command.call_args.args[0]["indexes"][0]["cosmosSearchOptions"]
    assert options["similarity"] == similarity


def test_custom_fields_round_trip():
    store = AzureDocumentDBDocumentStore(
        database_name="db", collection_name="collection", content_field="body", embedding_field="vector"
    )
    original = Document(id="id", content="content", embedding=[0.1], meta={"kind": "guide"})
    mongo_document = store._haystack_doc_to_mongo_doc(original)
    assert mongo_document["body"] == "content"
    assert mongo_document["vector"] == [0.1]
    assert store._mongo_doc_to_haystack_doc(mongo_document) == original


def test_default_credential_is_lazy(monkeypatch):
    monkeypatch.setenv("AZURE_DOCUMENTDB_CLUSTER_NAME", "cluster")
    with patch(
        "haystack_integrations.document_stores.azure_documentdb.document_store.DefaultAzureCredential"
    ) as credential_class:
        store = AzureDocumentDBDocumentStore(database_name="db", collection_name="collection")
        credential_class.assert_not_called()
        store._client_kwargs()
        credential_class.assert_called_once_with()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("database_name", "bad.name"), ("collection_name", "bad/name")],
)
def test_rejects_invalid_resource_names(parameter, value):
    kwargs = {"database_name": "db", "collection_name": "collection", parameter: value}
    with pytest.raises(ValueError, match=f"Invalid {parameter}"):
        AzureDocumentDBDocumentStore(**kwargs)


@pytest.mark.parametrize("parameter", ["embedding_field", "content_field"])
def test_rejects_invalid_field_names(parameter):
    kwargs = {"database_name": "db", "collection_name": "collection", parameter: "$invalid"}
    with pytest.raises(ValueError, match=parameter):
        AzureDocumentDBDocumentStore(**kwargs)


def test_crud_methods(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    collection.count_documents.return_value = 2
    collection.find.return_value = [{"id": "one", "content": "text", "meta": {}}]

    assert store.count_documents() == 2
    assert store.filter_documents({"field": "meta.kind", "operator": "==", "value": "guide"}) == [
        Document(id="one", content="text")
    ]
    store.delete_documents(["one"])

    collection.count_documents.assert_called_with({})
    collection.find.assert_called_once_with({"meta.kind": {"$eq": "guide"}})
    collection.delete_many.assert_called_once_with({"id": {"$in": ["one"]}})


def test_filter_mutation_methods(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    collection.delete_many.return_value.deleted_count = 2
    collection.update_many.return_value.modified_count = 3

    filters = {"field": "meta.kind", "operator": "==", "value": "guide"}
    assert store.delete_by_filter(filters) == 2
    assert store.update_by_filter(filters, {"reviewed": True}) == 3
    store.delete_all_documents()

    collection.delete_many.assert_any_call({"meta.kind": {"$eq": "guide"}})
    collection.update_many.assert_called_once_with({"meta.kind": {"$eq": "guide"}}, {"$set": {"meta.reviewed": True}})
    collection.delete_many.assert_any_call({})


@pytest.mark.parametrize(
    ("policy", "operation_type"),
    [
        (DuplicatePolicy.FAIL, InsertOne),
        (DuplicatePolicy.OVERWRITE, ReplaceOne),
        (DuplicatePolicy.SKIP, UpdateOne),
    ],
)
def test_write_documents_policies(mocked_store_collection, policy, operation_type):
    store, collection, _, _ = mocked_store_collection
    collection.count_documents.return_value = 0

    assert store.write_documents([Document(id="one", content="text")], policy=policy) == 1
    operation = collection.bulk_write.call_args.args[0][0]
    assert isinstance(operation, operation_type)


def test_write_documents_rejects_invalid_input(mocked_store_collection):
    store, _, _, _ = mocked_store_collection
    with pytest.raises(ValueError, match="objects of type Document"):
        store.write_documents(["not-a-document"])


def test_write_documents_translates_bulk_write_error(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    collection.bulk_write.side_effect = BulkWriteError({"writeErrors": [{"code": 11000}]})
    with pytest.raises(DuplicateDocumentError, match="Duplicate documents found"):
        store.write_documents([Document(id="one")])


def test_retrieval_validation_and_errors(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    with pytest.raises(ValueError, match="must not be empty"):
        store._embedding_retrieval([])
    with pytest.raises(ValueError, match="top_k"):
        store._embedding_retrieval([0.1], top_k=0)

    store.full_text_search_index = None
    with pytest.raises(ValueError, match="full_text_search_index"):
        store._full_text_retrieval("query")

    store.full_text_search_index = "text-index"
    collection.aggregate.side_effect = RuntimeError("query failed")
    with pytest.raises(DocumentStoreError, match="Vector retrieval"):
        store._embedding_retrieval([0.1])
    with pytest.raises(DocumentStoreError, match="Full-text retrieval"):
        store._full_text_retrieval("query")


def test_full_text_result_and_filters(mocked_store_collection):
    store, collection, _, _ = mocked_store_collection
    collection.aggregate.return_value = [{"id": "one", "content": "Azure", "meta": {}, "score": 1.5}]
    documents = store._full_text_retrieval(
        "azure", filters={"field": "meta.kind", "operator": "==", "value": "guide"}, top_k=1
    )
    pipeline = collection.aggregate.call_args.args[0]
    assert pipeline[1] == {"$match": {"meta.kind": {"$eq": "guide"}}}
    assert documents == [Document(id="one", content="Azure", score=1.5)]


def test_connection_validation_and_close(mocked_store_collection):
    store, _, client, _ = mocked_store_collection
    with pytest.raises(DocumentStoreError, match="not established"):
        _ = store.connection
    store.count_documents()
    assert store.connection is client
    store.close()
    client.close.assert_called_once_with()
    with pytest.raises(DocumentStoreError, match="not established"):
        _ = store.collection


def test_missing_collection_raises(mocked_store_collection):
    store, _, _, database = mocked_store_collection
    database.list_collection_names.return_value = []
    with pytest.raises(DocumentStoreError, match="does not exist"):
        store.count_documents()


def test_connection_failure_raises(mocked_store_collection):
    store, _, client, _ = mocked_store_collection
    client.admin.command.side_effect = RuntimeError("unreachable")
    with pytest.raises(DocumentStoreError, match="Connection to Azure DocumentDB failed"):
        store.count_documents()


@pytest.mark.skipif(
    not os.environ.get("AZURE_DOCUMENTDB_CONNECTION_STRING"), reason="No OSS DocumentDB connection string provided"
)
@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseExtendedTests):
    @pytest.fixture
    def document_store(self, real_collection):
        database_name, collection_name, _ = real_collection
        return AzureDocumentDBDocumentStore(database_name=database_name, collection_name=collection_name)

    def test_write_documents(self, document_store: AzureDocumentDBDocumentStore):
        documents = [Document(content="some text")]
        assert document_store.write_documents(documents) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents, DuplicatePolicy.FAIL)
