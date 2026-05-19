# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import random
from unittest.mock import AsyncMock, Mock, patch

import pytest
from elasticsearch.exceptions import BadRequestError  # type: ignore[import-not-found]
from haystack.dataclasses.document import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)
from haystack.utils import Secret
from haystack.utils.auth import TokenSecret

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_is_lazy(_mock_es_client):
    ElasticsearchDocumentStore(hosts="testhost")
    _mock_es_client.assert_not_called()


def test_init_with_special_fields_raises_error():
    with pytest.raises(ValueError, match=r"sparse_vector_field 'content' conflicts with a reserved field name\."):
        ElasticsearchDocumentStore(sparse_vector_field="content")


def test_init_ingest_pipeline_empty_raises():
    with pytest.raises(ValueError, match="ingest_pipeline must be a non-empty string"):
        ElasticsearchDocumentStore(ingest_pipeline="   ")


def test_init_ingest_pipeline_strips_whitespace():
    store = ElasticsearchDocumentStore(ingest_pipeline="  my-pipeline  ")
    assert store._ingest_pipeline == "my-pipeline"


def test_init_with_custom_mapping_injects_sparse_vector():
    custom_mapping = {"properties": {"some_field": {"type": "text"}}}
    store = ElasticsearchDocumentStore(custom_mapping=custom_mapping, sparse_vector_field="my_sparse_vec")
    assert "my_sparse_vec" in store._custom_mapping["properties"]
    assert store._custom_mapping["properties"]["my_sparse_vec"] == {"type": "sparse_vector"}


def test_handle_sparse_embedding_no_op_when_absent():
    store = ElasticsearchDocumentStore(hosts="testhost")
    doc_dict = {"id": "doc-1", "content": "hello"}
    store._handle_sparse_embedding(doc_dict, "doc-1")
    assert doc_dict == {"id": "doc-1", "content": "hello"}


def test_handle_sparse_embedding_converts_to_es_format():
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="my_sparse")
    doc_dict = {
        "id": "doc-1",
        "sparse_embedding": {"indices": [0, 5], "values": [0.3, 0.7]},
    }
    store._handle_sparse_embedding(doc_dict, "doc-1")
    assert "sparse_embedding" not in doc_dict
    assert doc_dict["my_sparse"] == {"0": 0.3, "5": 0.7}


def test_handle_sparse_embedding_warns_when_no_field_configured(caplog):
    store = ElasticsearchDocumentStore(hosts="testhost")
    doc_dict = {
        "id": "doc-1",
        "content": "hello",
        "sparse_embedding": {"indices": [0, 1], "values": [0.5, 0.5]},
    }
    store._handle_sparse_embedding(doc_dict, "doc-1")
    assert "but `sparse_vector_field` is not configured" in caplog.text
    assert "sparse_embedding" not in doc_dict


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_headers_are_supported(_mock_es_client):
    _ = ElasticsearchDocumentStore(
        hosts="http://testhost:9200", headers={"header1": "value1", "header2": "value2"}
    ).client

    assert _mock_es_client.call_count == 1
    _, kwargs = _mock_es_client.call_args

    headers_found = kwargs["headers"]
    assert headers_found["header1"] == "value1"
    assert headers_found["header2"] == "value2"
    assert headers_found["user-agent"].startswith("haystack-py-ds/")


def test_to_dict():
    document_store = ElasticsearchDocumentStore(hosts="some hosts")
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "api_key": {
                "env_vars": [
                    "ELASTIC_API_KEY",
                ],
                "strict": False,
                "type": "env_var",
            },
            "api_key_id": {
                "env_vars": [
                    "ELASTIC_API_KEY_ID",
                ],
                "strict": False,
                "type": "env_var",
            },
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "embedding_similarity_function": "cosine",
            "sparse_vector_field": None,
            "ingest_pipeline": None,
        },
    }


def test_from_dict():
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": None,
            "api_key_id": None,
            "embedding_similarity_function": "cosine",
            "sparse_vector_field": None,
            "ingest_pipeline": None,
        },
    }
    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._hosts == "some hosts"
    assert document_store._index == "default"
    assert document_store._custom_mapping is None
    assert document_store._api_key is None
    assert document_store._sparse_vector_field is None
    assert document_store._api_key_id is None
    assert document_store._embedding_similarity_function == "cosine"
    assert document_store._ingest_pipeline is None


def test_to_dict_with_api_keys_env_vars():
    document_store = ElasticsearchDocumentStore(hosts="https://localhost:9200")
    res = document_store.to_dict()
    assert res["init_parameters"]["api_key"] == {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False}
    assert res["init_parameters"]["api_key_id"] == {
        "type": "env_var",
        "env_vars": ["ELASTIC_API_KEY_ID"],
        "strict": False,
    }


def test_to_dict_with_api_keys_as_secret():
    with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
        document_store = ElasticsearchDocumentStore(
            hosts="https://localhost:9200",
            api_key=TokenSecret(_token="test-api-key"),
            api_key_id=TokenSecret(_token="test-api-key-id"),
        )
        _ = document_store.to_dict()


def test_to_dict_with_api_keys_str():
    document_store = ElasticsearchDocumentStore(
        hosts="https://localhost:9200", api_key="my_api_key", api_key_id="my_api_key_id"
    )
    res = document_store.to_dict()
    assert res["init_parameters"]["api_key"] is None
    assert res["init_parameters"]["api_key_id"] is None


def test_from_dict_with_api_keys_env_vars():
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False},
            "api_key_id": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY_ID"], "strict": False},
            "embedding_similarity_function": "cosine",
            "sparse_vector_field": None,
            "ingest_pipeline": None,
        },
    }

    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._api_key == Secret.from_env_var("ELASTIC_API_KEY", strict=False)
    assert document_store._api_key_id == Secret.from_env_var("ELASTIC_API_KEY_ID", strict=False)


def test_from_dict_with_api_keys_str():
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": "my_api_key",
            "api_key_id": "my_api_key_id",
            "embedding_similarity_function": "cosine",
            "sparse_vector_field": None,
            "ingest_pipeline": None,
        },
    }

    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._api_key == "my_api_key"
    assert document_store._api_key_id == "my_api_key_id"


def test_from_dict_without_sparse_vector_field():
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": "my_api_key",
            "api_key_id": "my_api_key_id",
            "embedding_similarity_function": "cosine",
        },
    }

    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._sparse_vector_field is None


def test_from_dict_without_ingest_pipeline():
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": "my_api_key",
            "api_key_id": "my_api_key_id",
            "embedding_similarity_function": "cosine",
            "sparse_vector_field": None,
        },
    }

    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._ingest_pipeline is None


@patch("haystack_integrations.document_stores.elasticsearch.document_store.helpers.bulk")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_write_documents_bulk_passes_pipeline_when_configured(mock_es, _mock_async_es, mock_bulk):
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    mock_client.indices.exists.return_value = True
    mock_es.return_value = mock_client
    mock_bulk.return_value = (1, [])

    store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        index="idx_test_pipeline",
        ingest_pipeline="my-ingest",
    )
    _ = store.client
    store.write_documents([Document(id="1", content="a")])

    mock_bulk.assert_called_once()
    assert mock_bulk.call_args.kwargs["pipeline"] == "my-ingest"

    call_actions = mock_bulk.call_args.kwargs["actions"]
    assert len(call_actions) == 1
    source = call_actions[0]["_source"]
    assert "embedding" not in source
    assert "blob" not in source
    assert "score" not in source


@patch("haystack_integrations.document_stores.elasticsearch.document_store.helpers.bulk")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_write_documents_bulk_omits_pipeline_when_not_configured(mock_es, _mock_async_es, mock_bulk):
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    mock_client.indices.exists.return_value = True
    mock_es.return_value = mock_client
    mock_bulk.return_value = (1, [])

    store = ElasticsearchDocumentStore(hosts="http://localhost:9200", index="idx_no_pipeline")
    _ = store.client
    store.write_documents([Document(id="1", content="a")])

    mock_bulk.assert_called_once()
    assert "pipeline" not in mock_bulk.call_args.kwargs


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.elasticsearch.document_store.helpers.async_bulk")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
async def test_write_documents_async_bulk_passes_pipeline_when_configured(mock_es, mock_async_es_cls, mock_async_bulk):
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    mock_client.indices.exists.return_value = True
    mock_es.return_value = mock_client

    mock_async_es_cls.return_value = AsyncMock()

    mock_async_bulk.return_value = (1, [])

    store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        index="idx_async_pipeline",
        ingest_pipeline="pipe-async",
    )
    _ = store.client
    await store.write_documents_async([Document(id="1", content="a")])

    mock_async_bulk.assert_called_once()
    assert mock_async_bulk.call_args.kwargs["pipeline"] == "pipe-async"


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.elasticsearch.document_store.helpers.async_bulk")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
async def test_write_documents_async_bulk_omits_pipeline_when_not_configured(
    mock_es, mock_async_es_cls, mock_async_bulk
):
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    mock_client.indices.exists.return_value = True
    mock_es.return_value = mock_client

    mock_async_es_cls.return_value = AsyncMock()

    mock_async_bulk.return_value = (1, [])

    store = ElasticsearchDocumentStore(hosts="http://localhost:9200", index="idx_async_no_pipeline")
    _ = store.client
    await store.write_documents_async([Document(id="1", content="a")])

    mock_async_bulk.assert_called_once()
    assert "pipeline" not in mock_async_bulk.call_args.kwargs


def test_api_key_validation_only_api_key():
    api_key = Secret.from_token("test_api_key")
    document_store = ElasticsearchDocumentStore(hosts="https://localhost:9200", api_key=api_key)
    assert document_store._api_key == api_key
    assert document_store._api_key_id == Secret.from_env_var("ELASTIC_API_KEY_ID", strict=False)


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
@patch.dict("os.environ", {"ELASTIC_API_KEY": "", "ELASTIC_API_KEY_ID": ""})
def test_api_key_validation_only_api_key_id_raises_error(_mock_elasticsearch_client):
    api_key_id = Secret.from_token("test_api_key_id")
    with pytest.raises(ValueError, match="api_key_id is provided but api_key is missing"):
        es = ElasticsearchDocumentStore(hosts="https://localhost:9200", api_key_id=api_key_id)
        es.client()


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
def test_client_initialization_with_api_key_tuple(_mock_async_es, _mock_es):
    api_key = Secret.from_token("test_api_key")
    api_key_id = Secret.from_token("test_api_key_id")

    # Mock the client.info() call to avoid actual connection
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    _mock_es.return_value = mock_client

    document_store = ElasticsearchDocumentStore(hosts="https://localhost:9200", api_key=api_key, api_key_id=api_key_id)

    # Access client to trigger initialization
    _ = document_store.client

    # Check that Elasticsearch was called with the correct api_key tuple
    _mock_es.assert_called_once()
    call_args = _mock_es.call_args
    assert call_args[0][0] == "https://localhost:9200"  # hosts
    assert call_args[1]["api_key"] == ("test_api_key_id", "test_api_key")

    # Check that AsyncElasticsearch was called with the same api_key tuple
    _mock_async_es.assert_called_once()
    async_call_args = _mock_async_es.call_args
    assert async_call_args[0][0] == "https://localhost:9200"  # hosts
    assert async_call_args[1]["api_key"] == ("test_api_key_id", "test_api_key")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
@patch("haystack_integrations.document_stores.elasticsearch.document_store.AsyncElasticsearch")
def test_client_initialization_with_api_key_string(_mock_async_es, _mock_es):
    api_key = "test_api_key"

    # Mock the client.info() call to avoid actual connection
    mock_client = Mock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    _mock_es.return_value = mock_client

    document_store = ElasticsearchDocumentStore(hosts="testhost", api_key=api_key)

    # Access client to trigger initialization
    _ = document_store.client

    # Check that Elasticsearch was called with the correct api_key string
    _mock_es.assert_called_once()
    call_args = _mock_es.call_args
    assert call_args[0][0] == "testhost"  # hosts
    assert call_args[1]["api_key"] == "test_api_key"

    # Check that AsyncElasticsearch was called with the same api_key string
    _mock_async_es.assert_called_once()
    async_call_args = _mock_async_es.call_args
    assert async_call_args[0][0] == "testhost"  # hosts
    assert async_call_args[1]["api_key"] == "test_api_key"


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_with_custom_mapping(mock_elasticsearch):
    custom_mapping = {
        "properties": {
            "embedding": {"type": "dense_vector", "index": True, "similarity": "dot_product"},
            "content": {"type": "text"},
        },
        "dynamic_templates": [
            {
                "strings": {
                    "path_match": "*",
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "keyword",
                    },
                }
            }
        ],
    }
    mock_client = Mock(
        indices=Mock(create=Mock(), exists=Mock(return_value=False)),
    )
    mock_elasticsearch.return_value = mock_client

    _ = ElasticsearchDocumentStore(hosts="http://testhost:9200", custom_mapping=custom_mapping).client
    mock_client.indices.create.assert_called_once_with(
        index="default",
        mappings=custom_mapping,
    )


def test_sparse_embedding_to_es_vector_mismatched_lengths():
    with pytest.raises(ValueError):
        ElasticsearchDocumentStore._sparse_embedding_to_es_vector(indices=[0, 1, 2], values=[0.5, 0.7])


def test_sparse_embedding_to_es_vector_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        ElasticsearchDocumentStore._sparse_embedding_to_es_vector(indices=[], values=[])


def test_deserialize_document_with_numeric_string_keys():
    # Documents written by Haystack use numeric string keys ("0", "1", ...)
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="sparse_vec")
    hit = {
        "_source": {"id": "doc-1", "content": "Berlin", "sparse_vec": {"0": 0.9, "2": 0.5, "1": 0.7}},
        "_score": 1.0,
    }
    doc = store._deserialize_document(hit)
    assert doc.sparse_embedding is not None
    assert doc.sparse_embedding.indices == [0, 1, 2]
    assert doc.sparse_embedding.values == [0.9, 0.7, 0.5]


def test_deserialize_document_with_elser_string_token_keys():
    # Documents indexed by an ES inference pipeline (ELSER) use token-string keys ("berlin", ...)
    # which cannot map to integer indices — sparse_embedding is left as None
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="sparse_vec")
    hit = {
        "_source": {"id": "doc-1", "content": "Berlin", "sparse_vec": {"berlin": 3.5, "capital": 2.1}},
        "_score": 1.0,
    }
    doc = store._deserialize_document(hit)
    assert doc.content == "Berlin"
    assert doc.score == 1.0
    assert doc.sparse_embedding is None


def test_deserialize_document_fields_api_with_numeric_keys():
    # When ES returns sparse data via the fields API (list-wrapped), numeric keys are deserialized correctly.
    # This covers the ingest-pipeline case where sparse_vec is absent from _source.
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="sparse_vec")
    hit = {
        "_source": {"id": "doc-1", "content": "Berlin"},
        "_score": 1.0,
        "fields": {"sparse_vec": [{"0": 0.9, "2": 0.5, "1": 0.7}]},
    }
    doc = store._deserialize_document(hit)
    assert doc.sparse_embedding is not None
    assert doc.sparse_embedding.indices == [0, 1, 2]
    assert doc.sparse_embedding.values == [0.9, 0.7, 0.5]


def test_deserialize_document_fields_api_takes_precedence_over_source():
    # fields API data wins over _source when both are present.
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="sparse_vec")
    hit = {
        "_source": {"id": "doc-1", "content": "Berlin", "sparse_vec": {"0": 0.1}},
        "_score": 1.0,
        "fields": {"sparse_vec": [{"0": 0.9, "1": 0.7}]},
    }
    doc = store._deserialize_document(hit)
    assert doc.sparse_embedding is not None
    assert doc.sparse_embedding.indices == [0, 1]
    assert doc.sparse_embedding.values == [0.9, 0.7]


def test_deserialize_document_fields_api_empty_list_does_not_crash():
    # ES can return the field key with an empty list if the field is mapped but has no value.
    store = ElasticsearchDocumentStore(hosts="testhost", sparse_vector_field="sparse_vec")
    hit = {
        "_source": {"id": "doc-1", "content": "Berlin"},
        "_score": 1.0,
        "fields": {"sparse_vec": []},
    }
    doc = store._deserialize_document(hit)
    assert doc.content == "Berlin"
    assert doc.sparse_embedding is None


def test_sparse_vector_retrieval_builds_query_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._sparse_vector_retrieval(
            query_sparse_embedding=SparseEmbedding(indices=[0, 2], values=[0.5, 0.7]),
            top_k=3,
        )

    mock_search.assert_called_once()
    search_kwargs = mock_search.call_args.kwargs
    assert search_kwargs["size"] == 3
    assert search_kwargs["query"]["bool"]["must"] == [
        {"sparse_vector": {"field": "sparse_vec", "query_vector": {"0": 0.5, "2": 0.7}}}
    ]
    assert "filter" not in search_kwargs["query"]["bool"]


def test_sparse_vector_retrieval_inference_builds_query_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._sparse_vector_retrieval_inference(query="Find Berlin", inference_id="ELSER", top_k=3)

    mock_search.assert_called_once()
    search_kwargs = mock_search.call_args.kwargs
    assert search_kwargs["size"] == 3
    assert search_kwargs["query"]["bool"]["must"] == [
        {"sparse_vector": {"field": "sparse_vec", "query": "Find Berlin", "inference_id": "ELSER"}}
    ]
    assert "filter" not in search_kwargs["query"]["bool"]


def test_sparse_vector_retrieval_inference_builds_query_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._sparse_vector_retrieval_inference(
            query="Find Berlin",
            inference_id="ELSER",
            filters={"field": "type", "operator": "==", "value": "match"},
            top_k=3,
        )

    mock_search.assert_called_once()
    search_kwargs = mock_search.call_args.kwargs
    assert search_kwargs["size"] == 3
    assert search_kwargs["query"]["bool"]["must"] == [
        {"sparse_vector": {"field": "sparse_vec", "query": "Find Berlin", "inference_id": "ELSER"}}
    ]
    assert search_kwargs["query"]["bool"]["filter"] == {"bool": {"must": {"term": {"type": "match"}}}}


def test_sparse_vector_retrieval_builds_query_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._sparse_vector_retrieval(
            query_sparse_embedding=SparseEmbedding(indices=[0, 2], values=[0.5, 0.7]),
            filters={"field": "type", "operator": "==", "value": "match"},
            top_k=3,
        )

    mock_search.assert_called_once()
    search_kwargs = mock_search.call_args.kwargs
    assert search_kwargs["size"] == 3
    assert search_kwargs["query"]["bool"]["must"] == [
        {"sparse_vector": {"field": "sparse_vec", "query_vector": {"0": 0.5, "2": 0.7}}}
    ]
    assert search_kwargs["query"]["bool"]["filter"] == {"bool": {"must": {"term": {"type": "match"}}}}


@pytest.mark.asyncio
async def test_sparse_vector_retrieval_async_builds_query_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._sparse_vector_retrieval_async(
        query_sparse_embedding=SparseEmbedding(indices=[1, 3], values=[0.4, 0.9]),
        top_k=2,
    )

    store._search_documents_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        size=2,
        query={
            "bool": {
                "must": [
                    {"sparse_vector": {"field": "sparse_vec", "query_vector": {"1": 0.4, "3": 0.9}}},
                ]
            }
        },
    )


@pytest.mark.asyncio
async def test_sparse_vector_retrieval_inference_async_builds_query_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._sparse_vector_retrieval_inference_async(
        query="Find Berlin",
        inference_id="ELSER",
        top_k=2,
    )

    store._search_documents_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        size=2,
        query={
            "bool": {
                "must": [
                    {"sparse_vector": {"field": "sparse_vec", "query": "Find Berlin", "inference_id": "ELSER"}},
                ]
            }
        },
    )


@pytest.mark.asyncio
async def test_sparse_vector_retrieval_inference_async_builds_query_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._sparse_vector_retrieval_inference_async(
        query="Find Berlin",
        inference_id="ELSER",
        filters={"field": "type", "operator": "==", "value": "match"},
        top_k=3,
    )

    store._search_documents_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        size=3,
        query={
            "bool": {
                "must": [
                    {"sparse_vector": {"field": "sparse_vec", "query": "Find Berlin", "inference_id": "ELSER"}},
                ],
                "filter": {"bool": {"must": {"term": {"type": "match"}}}},
            }
        },
    )


@pytest.mark.asyncio
async def test_sparse_vector_retrieval_async_builds_query_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._sparse_vector_retrieval_async(
        query_sparse_embedding=SparseEmbedding(indices=[1, 3], values=[0.4, 0.9]),
        filters={"field": "type", "operator": "==", "value": "match"},
        top_k=2,
    )

    store._search_documents_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        size=2,
        query={
            "bool": {
                "must": [
                    {"sparse_vector": {"field": "sparse_vec", "query_vector": {"1": 0.4, "3": 0.9}}},
                ],
                "filter": {"bool": {"must": {"term": {"type": "match"}}}},
            }
        },
    )


@pytest.mark.integration
class TestDocumentStore(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Common test cases will be provided by `DocumentStoreBaseExtendedTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["http://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        # this similarity function is rarely used in practice, but it is robust for test cases with fake embeddings
        # in fact, it works fine with vectors like [0.0] * 768, while cosine similarity would raise an exception
        embedding_similarity_function = "max_inner_product"

        store = ElasticsearchDocumentStore(
            hosts=hosts, index=index, embedding_similarity_function=embedding_similarity_function
        )
        yield store
        store.client.options(ignore_status=[400, 404]).indices.delete(index=index)
        store.client.close()

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        The ElasticSearchDocumentStore.filter_documents() method returns a Documents with their score set.
        We don't want to compare the score, so we set it to None before comparing the documents.
        """
        received_meta = []
        for doc in received:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            received_meta.append(r)

        expected_meta = []
        for doc in expected:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            expected_meta.append(r)
        received = [dataclasses.replace(doc, score=None) for doc in received]

        super().assert_documents_are_equal(received, expected)

    def test_user_agent_header(self, document_store: ElasticsearchDocumentStore):
        assert document_store.client._headers["user-agent"].startswith("haystack-py-ds/")

    def test_write_documents(self, document_store: ElasticsearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_documents_with_sparse_vectors(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_sync_sparse", sparse_vector_field="sparse_vec"
        )
        store.client.options(ignore_status=[400, 404]).indices.delete(index="test_sync_sparse")

        doc = Document(id="1", content="test", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5]))
        store.write_documents([doc])

        # check ES natively
        raw_doc = store.client.get(index="test_sync_sparse", id="1")
        assert raw_doc["_source"]["sparse_vec"] == {"0": 0.5, "1": 0.5}

        # check retrieval reconstruction
        results = store.filter_documents()
        assert len(results) == 1
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [0, 1]
        assert results[0].sparse_embedding.values == [0.5, 0.5]

        store.client.indices.delete(index="test_sync_sparse")

    def test_write_documents_with_non_contiguous_sparse_indices(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_sync_sparse_noncontiguous", sparse_vector_field="sparse_vec"
        )
        store.client.options(ignore_status=[400, 404]).indices.delete(index="test_sync_sparse_noncontiguous")

        doc = Document(
            id="1", content="test", sparse_embedding=SparseEmbedding(indices=[100, 5, 42], values=[0.1, 0.9, 0.5])
        )
        store.write_documents([doc])

        results = store.filter_documents()
        assert len(results) == 1
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [5, 42, 100]
        assert results[0].sparse_embedding.values == [0.9, 0.5, 0.1]

        store.client.indices.delete(index="test_sync_sparse_noncontiguous")

    def test_write_documents_mixed_sparse_and_non_sparse(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_sync_sparse_mixed", sparse_vector_field="sparse_vec"
        )
        store.client.options(ignore_status=[400, 404]).indices.delete(index="test_sync_sparse_mixed")

        docs = [
            Document(
                id="1", content="with sparse", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5])
            ),
            Document(id="2", content="without sparse"),
        ]
        store.write_documents(docs)

        results = sorted(store.filter_documents(), key=lambda d: d.id)
        assert len(results) == 2
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [0, 1]
        assert results[1].sparse_embedding is None

        store.client.indices.delete(index="test_sync_sparse_mixed")

    def test_write_documents_with_sparse_embedding_warning(self, document_store, caplog):
        """Test write_documents with document containing sparse_embedding field"""
        doc = Document(id="1", content="test", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5]))

        document_store.write_documents([doc])
        assert "but `sparse_vector_field` is not configured" in caplog.text

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].id == "1"
        assert not hasattr(results[0], "sparse_embedding") or results[0].sparse_embedding is None

    def test_bm25_retrieval(self, document_store: ElasticsearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = document_store._bm25_retrieval("functional", top_k=3)
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_pagination(self, document_store: ElasticsearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
                Document(content="Java is an object oriented programming language"),
                Document(content="Javascript is a programming language"),
                Document(content="Typescript is a programming language"),
                Document(content="C is a programming language"),
            ]
        )

        res = document_store._bm25_retrieval("programming", top_k=11)
        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    def test_bm25_retrieval_with_fuzziness(self, document_store: ElasticsearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        query_with_typo = "functinal"
        # Query without fuzziness to search for the exact match
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="0")
        # Nothing is found as the query contains a typo
        assert res == []

        # Query with fuzziness with the same query
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="1")
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_not_all_terms_must_match(self, document_store: ElasticsearchDocumentStore):
        """
        Test that not all terms must mandatorily match for BM25 retrieval to return a result.
        """
        documents = [
            Document(id="1", content="There are over 7,000 languages spoken around the world today."),
            Document(
                id="2",
                content=(
                    "Elephants have been observed to behave in a way that indicates a high level of self-awareness"
                    " such as recognizing themselves in mirrors."
                ),
            ),
            Document(
                id="3",
                content=(
                    "In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness"
                    " the phenomenon of bioluminescent waves."
                ),
            ),
        ]
        document_store.write_documents(documents)

        res = document_store._bm25_retrieval("How much self awareness do elephants have?", top_k=3)
        assert len(res) == 1
        assert res[0].id == "2"

    def test_embedding_retrieval(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store.write_documents(docs)
        results = document_store._embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    def test_embedding_retrieval_with_filters(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        results = document_store._embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters=filters)
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_pagination(self, document_store: ElasticsearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        docs = [
            Document(content=f"Document {i}", embedding=[random.random() for _ in range(4)])  # noqa: S311
            for i in range(20)
        ]

        document_store.write_documents(docs)
        results = document_store._embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=11, filters={})
        assert len(results) == 11

    def test_embedding_retrieval_query_documents_different_embedding_sizes(
        self, document_store: ElasticsearchDocumentStore
    ):
        """
        Test that the retrieval fails if the query embedding and the documents have different embedding sizes.
        """
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        document_store.write_documents(docs)

        with pytest.raises(BadRequestError):
            document_store._embedding_retrieval(query_embedding=[0.1, 0.1])

    def test_sparse_vector_retrieval_requires_sparse_vector_field(self, document_store: ElasticsearchDocumentStore):
        with pytest.raises(ValueError, match="sparse_vector_field must be set for sparse vector retrieval"):
            document_store._sparse_vector_retrieval(
                query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0])
            )

    def test_write_documents_different_embedding_sizes_fail(self, document_store: ElasticsearchDocumentStore):
        """
        Test that write_documents fails if the documents have different embedding sizes.
        """
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Hello world", embedding=[0.1, 0.2]),
        ]

        with pytest.raises(DocumentStoreError):
            document_store.write_documents(docs)

    def test_init_with_sparse_vector_field(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_init_sparse", sparse_vector_field="sparse_vec"
        )
        assert "sparse_vec" in store._default_mappings["properties"]
        assert store._default_mappings["properties"]["sparse_vec"]["type"] == "sparse_vector"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_init_with_custom_mapping(self, mock_elasticsearch):
        custom_mapping = {
            "properties": {
                "embedding": {"type": "dense_vector", "index": True, "similarity": "dot_product"},
                "content": {"type": "text"},
            },
            "dynamic_templates": [
                {
                    "strings": {
                        "path_match": "*",
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "keyword",
                        },
                    }
                }
            ],
        }
        mock_client = Mock(
            indices=Mock(create=Mock(), exists=Mock(return_value=False)),
        )
        mock_elasticsearch.return_value = mock_client

        _ = ElasticsearchDocumentStore(hosts="http://testhost:9200", custom_mapping=custom_mapping).client
        mock_client.indices.create.assert_called_once_with(
            index="default",
            mappings=custom_mapping,
        )

    def test_delete_all_documents_index_recreation(self, document_store: ElasticsearchDocumentStore):
        # populate the index with some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)

        # capture index structure before deletion
        assert document_store._client is not None
        index_info_before = document_store._client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

        # delete all documents
        document_store.delete_all_documents(recreate_index=True)
        assert document_store.count_documents() == 0

        # verify index structure is preserved
        index_info_after = document_store._client.indices.get(index=document_store._index)
        mappings_after = index_info_after[document_store._index]["mappings"]
        assert mappings_after == mappings_before, "delete_all_documents should preserve index mappings"

        settings_after = index_info_after[document_store._index]["settings"]
        settings_after["index"].pop("uuid", None)
        settings_after["index"].pop("creation_date", None)
        settings_before["index"].pop("uuid", None)
        settings_before["index"].pop("creation_date", None)
        assert settings_after == settings_before, "delete_all_documents should preserve index settings"

        # verify index can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    def test_count_unique_metadata_by_filter(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs)

        # test with only a subset of fields
        distinct_counts_subset = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["category", "status"]
        )
        assert distinct_counts_subset["category"] == 3
        assert distinct_counts_subset["status"] == 2
        assert "priority" not in distinct_counts_subset

        # test field name normalization (with "meta." prefix)
        distinct_counts_normalized = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["meta.category", "status", "meta.priority"]
        )
        assert distinct_counts_normalized["category"] == 3
        assert distinct_counts_normalized["status"] == 2
        assert distinct_counts_normalized["priority"] == 3

        # Test error handling when field doesn't exist
        with pytest.raises(ValueError, match="Fields not found in index mapping"):
            document_store.count_unique_metadata_by_filter(filters={}, metadata_fields=["nonexistent_field"])

    def test_get_metadata_fields_info(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"}),
        ]
        document_store.write_documents(docs)

        fields_info = document_store.get_metadata_fields_info()

        # verify that fields_info contains expected fields
        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info

        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"

    def test_get_metadata_field_unique_values(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "language": "Python"}),
            Document(content="Java programming", meta={"category": "B", "language": "Java"}),
            Document(content="Python scripting", meta={"category": "A", "language": "Python"}),
            Document(content="JavaScript development", meta={"category": "C", "language": "JavaScript"}),
            Document(content="Python data science", meta={"category": "A", "language": "Python"}),
            Document(content="Java backend", meta={"category": "B", "language": "Java"}),
        ]
        document_store.write_documents(docs)

        # test getting all unique values without search term
        unique_values, after_key = document_store.get_metadata_field_unique_values("meta.category", None, 10)
        assert set(unique_values) == {"A", "B", "C"}
        # after_key should be None when all results are returned
        assert after_key is None

        # Test with "meta." prefix
        unique_languages, _ = document_store.get_metadata_field_unique_values("meta.language", None, 10)
        assert set(unique_languages) == {"Python", "Java", "JavaScript"}

        # Test pagination - first page
        unique_values_page1, after_key_page1 = document_store.get_metadata_field_unique_values("meta.category", None, 2)
        assert len(unique_values_page1) == 2
        assert all(val in ["A", "B", "C"] for val in unique_values_page1)
        # Should have an after_key for pagination
        assert after_key_page1 is not None

        # Test pagination - second page using after_key
        unique_values_page2, after_key_page2 = document_store.get_metadata_field_unique_values(
            "meta.category", None, 2, after=after_key_page1
        )
        assert len(unique_values_page2) == 1
        assert unique_values_page2[0] in ["A", "B", "C"]
        # Should have no more results
        assert after_key_page2 is None

        # Test with search term - filter by content matching "Python"
        unique_values_filtered, _ = document_store.get_metadata_field_unique_values("meta.category", "Python", 10)
        assert set(unique_values_filtered) == {"A"}  # Only category A has documents with "Python" in content

        # Test with search term - filter by content matching "Java"
        unique_values_java, _ = document_store.get_metadata_field_unique_values("meta.category", "Java", 10)
        assert set(unique_values_java) == {"B"}  # Only category B has documents with "Java" in content

        # Test with integer values
        int_docs = [
            Document(content="Doc 1", meta={"priority": 1}),
            Document(content="Doc 2", meta={"priority": 2}),
            Document(content="Doc 3", meta={"priority": 1}),
            Document(content="Doc 4", meta={"priority": 3}),
        ]
        document_store.write_documents(int_docs)
        unique_priorities, _ = document_store.get_metadata_field_unique_values("meta.priority", None, 10)
        assert set(unique_priorities) == {"1", "2", "3"}

        # Test with search term on integer field
        unique_priorities_filtered, _ = document_store.get_metadata_field_unique_values("meta.priority", "Doc 1", 10)
        assert set(unique_priorities_filtered) == {"1"}

    def test_query_sql(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
            Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
        ]
        document_store.write_documents(docs)

        # SQL query returns raw JSON response from Elasticsearch SQL API
        sql_query = (
            f'SELECT content, category, status, priority FROM "{document_store._index}" '  # noqa: S608
            f"WHERE category = 'A' ORDER BY priority"
        )
        result = document_store._query_sql(sql_query)

        # Verify raw JSON response structure
        assert isinstance(result, dict)
        assert "columns" in result
        assert "rows" in result

        # Verify we got 2 rows (documents with category A)
        assert len(result["rows"]) == 2

        # Verify column structure
        column_names = [col["name"] for col in result["columns"]]
        assert "content" in column_names
        assert "category" in column_names

    def test_query_sql_with_fetch_size(self, document_store: ElasticsearchDocumentStore):
        """Test SQL query with fetch_size parameter"""
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
        document_store.write_documents(docs)

        sql_query = (
            f'SELECT content, category FROM "{document_store._index}" '  # noqa: S608
            f"WHERE category = 'A'"
        )

        # Test with fetch_size
        result = document_store._query_sql(sql_query, fetch_size=5)

        # Should return raw JSON response
        assert isinstance(result, dict)
        assert "columns" in result
        assert "rows" in result

    def test_query_sql_error_handling(self, document_store: ElasticsearchDocumentStore):
        """Test error handling for invalid SQL queries"""

        invalid_query = "SELECT * FROM non_existent_index"
        with pytest.raises(DocumentStoreError, match="Failed to execute SQL query"):
            document_store._query_sql(invalid_query)


def test_hybrid_retrieval_inference_builds_body_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._hybrid_retrieval_inference(query="Find Berlin", inference_id="ELSER", top_k=3)

    mock_search.assert_called_once()
    body = mock_search.call_args.kwargs
    assert body["size"] == 3
    rrf = body["retriever"]["rrf"]
    assert rrf["rank_window_size"] == 100
    assert rrf["rank_constant"] == 60
    bm25, sparse = rrf["retrievers"]
    assert bm25["standard"]["query"]["multi_match"]["query"] == "Find Berlin"
    assert bm25["standard"]["query"]["multi_match"]["fuzziness"] == "AUTO"
    assert "filter" not in bm25["standard"]
    assert sparse["standard"]["query"]["sparse_vector"]["field"] == "sparse_vec"
    assert sparse["standard"]["query"]["sparse_vector"]["inference_id"] == "ELSER"
    assert sparse["standard"]["query"]["sparse_vector"]["query"] == "Find Berlin"
    assert "filter" not in sparse["standard"]


def test_hybrid_retrieval_inference_builds_body_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._hybrid_retrieval_inference(
            query="Find Berlin",
            inference_id="ELSER",
            filters={"field": "type", "operator": "==", "value": "match"},
            top_k=3,
        )

    mock_search.assert_called_once()
    body = mock_search.call_args.kwargs
    rrf = body["retriever"]["rrf"]
    bm25, sparse = rrf["retrievers"]
    expected_filter = {"bool": {"must": {"term": {"type": "match"}}}}
    assert bm25["standard"]["filter"] == expected_filter
    assert sparse["standard"]["filter"] == expected_filter


def test_hybrid_retrieval_inference_builds_body_custom_rank_params():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")

    with patch.object(store, "_search_documents", return_value=[]) as mock_search:
        store._hybrid_retrieval_inference(
            query="Find Berlin",
            inference_id="ELSER",
            rank_window_size=50,
            rank_constant=20,
        )

    rrf = mock_search.call_args.kwargs["retriever"]["rrf"]
    assert rrf["rank_window_size"] == 50
    assert rrf["rank_constant"] == 20


def test_hybrid_retrieval_inference_raises_without_sparse_field():
    store = ElasticsearchDocumentStore(hosts="some hosts")
    with pytest.raises(ValueError, match="sparse_vector_field must be set"):
        store._hybrid_retrieval_inference(query="test", inference_id="ELSER")


def test_hybrid_retrieval_inference_raises_on_empty_query():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store._hybrid_retrieval_inference(query="", inference_id="ELSER")


@pytest.mark.asyncio
async def test_hybrid_retrieval_inference_async_builds_body_without_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._hybrid_retrieval_inference_async(query="Find Berlin", inference_id="ELSER", top_k=3)

    store._search_documents_async.assert_awaited_once()
    body = store._search_documents_async.call_args.kwargs
    assert body["size"] == 3
    rrf = body["retriever"]["rrf"]
    bm25, sparse = rrf["retrievers"]
    assert bm25["standard"]["query"]["multi_match"]["query"] == "Find Berlin"
    assert sparse["standard"]["query"]["sparse_vector"]["inference_id"] == "ELSER"
    assert "filter" not in bm25["standard"]
    assert "filter" not in sparse["standard"]


@pytest.mark.asyncio
async def test_hybrid_retrieval_inference_async_builds_body_with_filters():
    store = ElasticsearchDocumentStore(hosts="some hosts", sparse_vector_field="sparse_vec")
    store._initialized = True
    store._search_documents_async = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await store._hybrid_retrieval_inference_async(
        query="Find Berlin",
        inference_id="ELSER",
        filters={"field": "type", "operator": "==", "value": "match"},
        top_k=3,
    )

    store._search_documents_async.assert_awaited_once()
    body = store._search_documents_async.call_args.kwargs
    rrf = body["retriever"]["rrf"]
    bm25, sparse = rrf["retrievers"]
    expected_filter = {"bool": {"must": {"term": {"type": "match"}}}}
    assert bm25["standard"]["filter"] == expected_filter
    assert sparse["standard"]["filter"] == expected_filter
