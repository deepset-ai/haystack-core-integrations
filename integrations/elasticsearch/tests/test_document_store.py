# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
import time
from typing import List
from unittest.mock import Mock, patch

import pytest
from elasticsearch.exceptions import BadRequestError  # type: ignore[import-not-found]
from haystack.dataclasses.document import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret
from haystack.utils.auth import TokenSecret

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_is_lazy(_mock_es_client):
    ElasticsearchDocumentStore(hosts="testhost")
    _mock_es_client.assert_not_called()


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


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
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
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": None,
            "api_key_id": None,
            "embedding_similarity_function": "cosine",
        },
    }
    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._hosts == "some hosts"
    assert document_store._index == "default"
    assert document_store._custom_mapping is None
    assert document_store._api_key is None
    assert document_store._api_key_id is None
    assert document_store._embedding_similarity_function == "cosine"


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict_with_api_keys_env_vars(_mock_elasticsearch_client, monkeypatch):
    monkeypatch.setenv("ELASTIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ELASTIC_API_KEY_ID", "test-api-key-id")
    document_store = ElasticsearchDocumentStore(hosts="https://localhost:9200")
    document_store.client()
    res = document_store.to_dict()
    assert res["init_parameters"]["api_key"] == {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False}
    assert res["init_parameters"]["api_key_id"] == {
        "type": "env_var",
        "env_vars": ["ELASTIC_API_KEY_ID"],
        "strict": False,
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict_with_api_keys_as_secret(_mock_elasticsearch_client, monkeypatch):
    monkeypatch.setenv("ELASTIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ELASTIC_API_KEY_ID", "test-api-key-id")
    with pytest.raises(ValueError):
        document_store = ElasticsearchDocumentStore(
            hosts="https://localhost:9200",
            api_key=TokenSecret(_token="test-api-key"),
            api_key_id=TokenSecret(_token="test-api-key-id"),
        )
        document_store.client()
        _ = document_store.to_dict()


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_with_api_keys_env_vars(_mock_elasticsearch_client):
    data = {
        "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "custom_mapping": None,
            "index": "default",
            "api_key": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False},
            "api_key_id": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY_ID"], "strict": False},
            "embedding_similarity_function": "cosine",
        },
    }

    document_store = ElasticsearchDocumentStore.from_dict(data)
    assert document_store._api_key == {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False}
    assert document_store._api_key_id == {"type": "env_var", "env_vars": ["ELASTIC_API_KEY_ID"], "strict": False}


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_api_key_validation_only_api_key(_mock_elasticsearch_client):
    api_key = Secret.from_token("test_api_key")

    document_store = ElasticsearchDocumentStore(hosts="https://localhost:9200", api_key=api_key)
    document_store.client()
    assert document_store._api_key == api_key
    # not passing the api_key_id makes it default to reading from env var
    assert document_store._api_key_id == Secret.from_env_var("ELASTIC_API_KEY_ID", strict=False)


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
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
    api_key = Secret.from_token("test_api_key")

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


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
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

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
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
        for doc in received:
            doc.score = None

        super().assert_documents_are_equal(received, expected)

    def test_user_agent_header(self, document_store: ElasticsearchDocumentStore):
        assert document_store.client._headers["user-agent"].startswith("haystack-py-ds/")

    def test_write_documents(self, document_store: ElasticsearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

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

    def test_delete_all_documents_no_index_recreation(self, document_store: ElasticsearchDocumentStore):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents(recreate_index=False)
        time.sleep(2)  # need to wait for the deletion to be reflected in count_documents
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"


@pytest.mark.integration
class TestElasticsearchDocumentStoreAsync:
    @pytest.fixture
    async def document_store(self, request):
        """
        Basic fixture providing a document store instance for async tests
        """
        hosts = ["http://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = ElasticsearchDocumentStore(hosts=hosts, index=index)
        yield store
        store.client.options(ignore_status=[400, 404]).indices.delete(index=index)

        await store.async_client.close()

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store):
        docs = [Document(id="1", content="test")]
        assert await document_store.write_documents_async(docs) == 1
        assert await document_store.count_documents_async() == 1
        with pytest.raises(DocumentStoreError):
            await document_store.write_documents_async(docs, policy=DuplicatePolicy.FAIL)

    @pytest.mark.asyncio
    async def test_count_documents_async(self, document_store):
        docs = [
            Document(content="test doc 1"),
            Document(content="test doc 2"),
            Document(content="test doc 3"),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1
        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, document_store):
        filterable_docs = [
            Document(content="1", meta={"number": -10}),
            Document(content="2", meta={"number": 100}),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "number", "operator": "==", "value": 100}
        )
        assert len(result) == 1
        assert result[0].meta["number"] == 100

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async(self, document_store):
        docs = [
            Document(content="Haskell is a functional programming language"),
            Document(content="Python is an object oriented programming language"),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._bm25_retrieval_async("functional", top_k=1)
        assert len(results) == 1
        assert "functional" in results[0].content

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, document_store):
        # init document store
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Less similar document", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        await document_store.write_documents_async(docs)

        # without num_candidates set to None
        results = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].content == "Most similar document"

        # with num_candidates not None
        results = await document_store._embedding_retrieval_async(
            query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=2, num_candidates=2
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"

        # with an embedding containing None
        with pytest.raises(ValueError, match="query_embedding must be a non-empty list of floats"):
            _ = await document_store._embedding_retrieval_async(query_embedding=None, top_k=2)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async_with_filters(self, document_store):
        docs = [
            Document(content="Haskell is a functional programming language", meta={"type": "functional"}),
            Document(content="Python is an object oriented programming language", meta={"type": "oop"}),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._bm25_retrieval_async(
            "programming", filters={"field": "type", "operator": "==", "value": "functional"}, top_k=1
        )
        assert len(results) == 1
        assert "functional" in results[0].content

        # test with scale_score=True
        results = await document_store._bm25_retrieval_async(
            "programming", filters={"field": "type", "operator": "==", "value": "functional"}, top_k=1, scale_score=True
        )
        assert len(results) == 1
        assert "functional" in results[0].content
        assert 0 <= results[0].score <= 1  # score should be between 0 and 1

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_with_filters(self, document_store):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0], meta={"type": "similar"}),
            Document(content="Less similar document", embedding=[0.5, 0.5, 0.5, 0.5], meta={"type": "different"}),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._embedding_retrieval_async(
            query_embedding=[1.0, 1.0, 1.0, 1.0],
            filters={"field": "type", "operator": "==", "value": "similar"},
            top_k=1,
        )
        assert len(results) == 1
        assert results[0].content == "Most similar document"

    @pytest.mark.asyncio
    async def test_write_documents_async_invalid_document_type(self, document_store):
        """Test write_documents with invalid document type"""
        invalid_docs = [{"id": "1", "content": "test"}]  # Dictionary instead of Document object
        with pytest.raises(ValueError, match="param 'documents' must contain a list of objects of type Document"):
            await document_store.write_documents_async(invalid_docs)

    @pytest.mark.asyncio
    async def test_write_documents_async_with_sparse_embedding_warning(self, document_store, caplog):
        """Test write_documents with document containing sparse_embedding field"""
        doc = Document(id="1", content="test", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5]))

        await document_store.write_documents_async([doc])
        assert "but storing sparse embeddings in Elasticsearch is not currently supported." in caplog.text

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "1"
        assert not hasattr(results[0], "sparse_embedding") or results[0].sparse_embedding is None

    @pytest.mark.asyncio
    async def test_delete_all_documents_async(self, document_store):
        docs = [
            Document(id="1", content="First document", meta={"category": "test"}),
            Document(id="2", content="Second document", meta={"category": "test"}),
            Document(id="3", content="Third document", meta={"category": "other"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # delete all documents
        await document_store.delete_all_documents_async(recreate_index=False)
        assert await document_store.count_documents_async() == 0

        # verify index still exists and can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "4"
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_index_recreation(self, document_store):
        # populate the index with some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)

        # capture index structure before deletion
        assert document_store._async_client is not None
        index_info_before = await document_store._async_client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

        # delete all documents with index recreation
        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

        # verify index structure is preserved
        index_info_after = await document_store._async_client.indices.get(index=document_store._index)
        mappings_after = index_info_after[document_store._index]["mappings"]
        assert mappings_after == mappings_before, "delete_all_documents_async should preserve index mappings"

        settings_after = index_info_after[document_store._index]["settings"]
        settings_after["index"].pop("uuid", None)
        settings_after["index"].pop("creation_date", None)
        settings_before["index"].pop("uuid", None)
        settings_before["index"].pop("creation_date", None)
        assert settings_after == settings_before, "delete_all_documents_async should preserve index settings"

        # verify index can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_no_index_recreation(self, document_store):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(recreate_index=False)
        # Need to wait for the deletion to be reflected in count_documents
        time.sleep(2)
        assert await document_store.count_documents_async() == 0

        new_doc = Document(id="3", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1
