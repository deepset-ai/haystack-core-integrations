# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import Mock, patch

import pytest
from elasticsearch.exceptions import BadRequestError  # type: ignore[import-not-found]
from haystack.dataclasses.document import Document
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

        document_store.delete_all_documents(recreate_index=False, refresh=True)
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    def test_delete_by_filter(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Delete documents with category="A"
        deleted_count = document_store.delete_by_filter(
            filters={"field": "category", "operator": "==", "value": "A"}, refresh=True
        )
        assert deleted_count == 2
        assert document_store.count_documents() == 1

        # Verify only category B remains
        remaining_docs = document_store.filter_documents()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    def test_update_by_filter(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Update status for category="A" documents
        updated_count = document_store.update_by_filter(
            filters={"field": "category", "operator": "==", "value": "A"},
            meta={"status": "published"},
            refresh=True,
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = document_store.filter_documents(
            filters={"field": "status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = document_store.filter_documents(filters={"field": "status", "operator": "==", "value": "draft"})
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"

    def test_count_documents_by_filter(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)

        count_a = document_store.count_documents_by_filter(
            filters={"field": "category", "operator": "==", "value": "A"}
        )
        assert count_a == 3

        count_a_active = document_store.count_documents_by_filter(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "category", "operator": "==", "value": "A"},
                    {"field": "status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count_a_active == 2

    def test_count_unique_metadata_by_filter(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs)

        # count distinct values for all documents
        distinct_counts = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["category", "status", "priority"]
        )
        assert distinct_counts["category"] == 3  # A, B, C
        assert distinct_counts["status"] == 2  # active, inactive
        assert distinct_counts["priority"] == 3  # 1, 2, 3

        # distinct values for documents with category="A"
        distinct_counts_a = document_store.count_unique_metadata_by_filter(
            filters={"field": "category", "operator": "==", "value": "A"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a["category"] == 1  # Only A
        assert distinct_counts_a["status"] == 2  # active, inactive
        assert distinct_counts_a["priority"] == 2  # 1, 3

        # distinct values for documents with status="active"
        distinct_counts_active = document_store.count_unique_metadata_by_filter(
            filters={"field": "status", "operator": "==", "value": "active"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_active["category"] == 3  # A, B, C
        assert distinct_counts_active["status"] == 1  # Only active
        assert distinct_counts_active["priority"] == 3  # 1, 2, 3

        # distinct values with complex filter (category="A" AND status="active")
        distinct_counts_a_active = document_store.count_unique_metadata_by_filter(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "category", "operator": "==", "value": "A"},
                    {"field": "status", "operator": "==", "value": "active"},
                ],
            },
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a_active["category"] == 1  # Only A
        assert distinct_counts_a_active["status"] == 1  # Only active
        assert distinct_counts_a_active["priority"] == 2  # 1, 3

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

    def test_get_metadata_field_min_max(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"priority": 1, "age": 10}),
            Document(content="Doc 2", meta={"priority": 5, "age": 20}),
            Document(content="Doc 3", meta={"priority": 3, "age": 15}),
            Document(content="Doc 4", meta={"priority": 10, "age": 5}),
            Document(content="Doc 6", meta={"rating": 10.5}),
            Document(content="Doc 7", meta={"rating": 20.3}),
            Document(content="Doc 8", meta={"rating": 15.7}),
            Document(content="Doc 9", meta={"rating": 5.2}),
        ]
        document_store.write_documents(docs)

        # test with "meta." prefix for integer field
        min_max_priority = document_store.get_metadata_field_min_max("meta.priority")
        assert min_max_priority["min"] == 1
        assert min_max_priority["max"] == 10

        # test with single value
        single_doc = [Document(content="Doc 5", meta={"single_value": 42})]
        document_store.write_documents(single_doc)
        min_max_single = document_store.get_metadata_field_min_max("meta.single_value")
        assert min_max_single["min"] == 42
        assert min_max_single["max"] == 42

        # test with float values
        min_max_score = document_store.get_metadata_field_min_max("meta.rating")
        assert min_max_score["min"] == pytest.approx(5.2)
        assert min_max_score["max"] == pytest.approx(20.3)

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
