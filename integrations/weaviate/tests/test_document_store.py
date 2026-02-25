# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from dateutil import parser
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.testing.document_store import (
    DocumentStoreBaseExtendedTests,
    create_filterable_docs,
)
from haystack.utils.auth import Secret
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import float32 as np_float32
from weaviate.collections.classes.data import DataObject
from weaviate.config import AdditionalConfig, ConnectionConfig, Proxies, Timeout
from weaviate.embedded import (
    DEFAULT_BINARY_PATH,
    DEFAULT_GRPC_PORT,
    DEFAULT_PERSISTENCE_DATA_PATH,
    DEFAULT_PORT,
    EmbeddedOptions,
)

from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate.WeaviateClient")
def test_init_is_lazy(_mock_client):
    _ = WeaviateDocumentStore()
    _mock_client.assert_not_called()


@pytest.mark.integration
class TestWeaviateDocumentStore(DocumentStoreBaseExtendedTests):
    @pytest.fixture
    def document_store(self, request) -> WeaviateDocumentStore:
        # Use a different index for each test so we can run them in parallel
        collection_settings = {
            "class": f"{request.node.name}",
            "invertedIndexConfig": {"indexNullState": True},
            "properties": [
                *DOCUMENT_COLLECTION_PROPERTIES,
                {"name": "number", "dataType": ["int"]},
                {"name": "date", "dataType": ["date"]},
                {"name": "category", "dataType": ["text"]},
                {"name": "status", "dataType": ["text"]},
            ],
        }
        store = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        yield store
        store.client.collections.delete(collection_settings["class"])

    @pytest.fixture
    def filterable_docs(self) -> list[Document]:
        """
        This fixture has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Weaviate forces RFC 3339 date strings.
        The original fixture uses ISO 8601 date strings.
        """
        documents = create_filterable_docs()
        for i in range(len(documents)):
            if date := documents[i].meta.get("date"):
                documents[i].meta["date"] = f"{date}Z"
        return documents

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        assert len(received) == len(expected)
        received = sorted(received, key=lambda doc: doc.id)
        expected = sorted(expected, key=lambda doc: doc.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc_dict = received_doc.to_dict(flatten=False)
            expected_doc_dict = expected_doc.to_dict(flatten=False)

            # Weaviate stores embeddings with lower precision floats so we handle that here.
            assert np_array_equal(
                np_array(received_doc_dict.pop("embedding", None), dtype=np_float32),
                np_array(expected_doc_dict.pop("embedding", None), dtype=np_float32),
                equal_nan=True,
            )

            received_meta = received_doc_dict.pop("meta", None)
            expected_meta = expected_doc_dict.pop("meta", None)

            assert received_doc_dict == expected_doc_dict

            # If a meta field is not set in a saved document, it will be None when retrieved
            # from Weaviate so we need to handle that.
            meta_keys = set(received_meta.keys()).union(set(expected_meta.keys()))
            for key in meta_keys:
                assert received_meta.get(key) == expected_meta.get(key)

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate.WeaviateClient")
    def test_connection(self, mock_weaviate_client_class, monkeypatch):
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_weaviate_client_class.return_value = mock_client
        monkeypatch.setenv("WEAVIATE_API_KEY", "my_api_key")
        ds = WeaviateDocumentStore(
            collection_settings={"class": "My_collection"},
            auth_client_secret=AuthApiKey(),
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.7",
                hostname="127.0.0.1",
            ),
            additional_config=AdditionalConfig(
                proxies={"http": "http://proxy:1234"}, trust_env=False, timeout=(10, 60)
            ),
        )

        # Trigger the actual database connection by accessing the `client` property so we
        # can assert the setup was good
        _ = ds.client

        # Verify client is created with correct parameters
        mock_weaviate_client_class.assert_called_once_with(
            auth_client_secret=AuthApiKey().resolve_value(),
            connection_params=None,
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.7",
                hostname="127.0.0.1",
            ),
            skip_init_checks=False,
            additional_config=AdditionalConfig(
                proxies={"http": "http://proxy:1234"}, trust_env=False, timeout=(10, 60)
            ),
        )

        # Verify collection is created
        mock_client.collections.exists.assert_called_once_with("My_collection")
        mock_client.collections.create_from_dict.assert_called_once_with(
            {"class": "My_collection", "properties": DOCUMENT_COLLECTION_PROPERTIES}
        )

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
    def test_to_dict(self, _mock_weaviate, monkeypatch):
        monkeypatch.setenv("WEAVIATE_API_KEY", "my_api_key")
        document_store = WeaviateDocumentStore(
            url="http://localhost:8080",
            auth_client_secret=AuthApiKey(),
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.0",
                hostname="127.0.0.1",
            ),
            additional_config=AdditionalConfig(
                connection=ConnectionConfig(),
                timeout=(30, 90),
                trust_env=False,
                proxies={"http": "http://proxy:1234"},
            ),
        )
        assert document_store.to_dict() == {
            "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
            "init_parameters": {
                "url": "http://localhost:8080",
                "collection_settings": {
                    "class": "Default",
                    "invertedIndexConfig": {"indexNullState": True},
                    "properties": [
                        {"name": "_original_id", "dataType": ["text"]},
                        {"name": "content", "dataType": ["text"]},
                        {"name": "blob_data", "dataType": ["blob"]},
                        {"name": "blob_mime_type", "dataType": ["text"]},
                        {"name": "score", "dataType": ["number"]},
                    ],
                },
                "auth_client_secret": {
                    "type": "api_key",
                    "init_parameters": {
                        "api_key": {"env_vars": ["WEAVIATE_API_KEY"], "strict": True, "type": "env_var"}
                    },
                },
                "additional_headers": {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
                "embedded_options": {
                    "persistence_data_path": DEFAULT_PERSISTENCE_DATA_PATH,
                    "binary_path": DEFAULT_BINARY_PATH,
                    "version": "1.23.0",
                    "port": DEFAULT_PORT,
                    "hostname": "127.0.0.1",
                    "additional_env_vars": None,
                    "grpc_port": DEFAULT_GRPC_PORT,
                },
                "additional_config": {
                    "connection": {
                        "session_pool_connections": 20,
                        "session_pool_maxsize": 100,
                        "session_pool_max_retries": 3,
                        "session_pool_timeout": 5,
                    },
                    "grpc_config": None,
                    "proxies": {"http": "http://proxy:1234", "https": None, "grpc": None},
                    "timeout": [30, 90],
                    "trust_env": False,
                },
            },
        }

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
    def test_from_dict(self, _mock_weaviate, monkeypatch):
        monkeypatch.setenv("WEAVIATE_API_KEY", "my_api_key")
        document_store = WeaviateDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                "init_parameters": {
                    "url": "http://localhost:8080",
                    "collection_settings": None,
                    "auth_client_secret": {
                        "type": "api_key",
                        "init_parameters": {
                            "api_key": {"env_vars": ["WEAVIATE_API_KEY"], "strict": True, "type": "env_var"}
                        },
                    },
                    "additional_headers": {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
                    "embedded_options": {
                        "persistence_data_path": DEFAULT_PERSISTENCE_DATA_PATH,
                        "binary_path": DEFAULT_BINARY_PATH,
                        "version": "1.23.0",
                        "port": DEFAULT_PORT,
                        "hostname": "127.0.0.1",
                        "additional_env_vars": None,
                        "grpc_port": DEFAULT_GRPC_PORT,
                    },
                    "additional_config": {
                        "connection": {
                            "session_pool_connections": 20,
                            "session_pool_maxsize": 20,
                            "session_pool_timeout": 5,
                        },
                        "proxies": {"http": "http://proxy:1234"},
                        "timeout": [10, 60],
                        "trust_env": False,
                    },
                },
            }
        )

        assert document_store._url == "http://localhost:8080"
        assert document_store._collection_settings == {
            "class": "Default",
            "invertedIndexConfig": {"indexNullState": True},
            "properties": [
                {"name": "_original_id", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "blob_data", "dataType": ["blob"]},
                {"name": "blob_mime_type", "dataType": ["text"]},
                {"name": "score", "dataType": ["number"]},
            ],
        }
        assert document_store._auth_client_secret == AuthApiKey()
        assert document_store._additional_config.timeout == Timeout(query=10, insert=60)
        assert document_store._additional_config.proxies == Proxies(http="http://proxy:1234", https=None, grpc=None)
        assert not document_store._additional_config.trust_env
        assert document_store._additional_headers == {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"}
        assert document_store._embedded_options.persistence_data_path == DEFAULT_PERSISTENCE_DATA_PATH
        assert document_store._embedded_options.binary_path == DEFAULT_BINARY_PATH
        assert document_store._embedded_options.version == "1.23.0"
        assert document_store._embedded_options.port == DEFAULT_PORT
        assert document_store._embedded_options.hostname == "127.0.0.1"
        assert document_store._embedded_options.additional_env_vars is None
        assert document_store._embedded_options.grpc_port == DEFAULT_GRPC_PORT
        assert document_store._additional_config.connection.session_pool_connections == 20
        assert document_store._additional_config.connection.session_pool_maxsize == 20
        assert document_store._additional_config.connection.session_pool_timeout == 5

    def test_to_data_object(self, test_files_path):
        doc = Document(content="test doc")
        data = WeaviateDocumentStore._to_data_object(doc)
        assert data == {
            "_original_id": doc.id,
            "content": doc.content,
            "score": None,
        }

        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(
            content="test doc",
            blob=image,
            embedding=[1, 2, 3],
            meta={"key": "value"},
        )
        data = WeaviateDocumentStore._to_data_object(doc)
        assert data == {
            "_original_id": doc.id,
            "content": doc.content,
            "blob_data": base64.b64encode(image.data).decode(),
            "blob_mime_type": "image/jpeg",
            "score": None,
            "key": "value",
        }

    def test_to_document(self, test_files_path):
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        data = DataObject(
            properties={
                "_original_id": "123",
                "content": "some content",
                "blob_data": base64.b64encode(image.data).decode(),
                "blob_mime_type": "image/jpeg",
                "score": None,
                "key": "value",
            },
            vector={"default": [1, 2, 3]},
        )

        doc = WeaviateDocumentStore._to_document(data)
        assert doc.id == "123"
        assert doc.content == "some content"
        assert doc.blob == image
        assert doc.embedding == [1, 2, 3]
        assert doc.score is None
        assert doc.meta == {"key": "value"}

    def test_write_documents(self, document_store):
        """
        Test write_documents() with default policy overwrites existing documents.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        assert document_store.count_documents() == 1

        doc.content = "test doc 2"
        assert document_store.write_documents([doc]) == 1
        assert document_store.count_documents() == 1

    def test_write_documents_with_blob_data(self, document_store, test_files_path):
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(content="test doc", blob=image)
        assert document_store.write_documents([doc]) == 1

    def test_filter_documents_with_blob_data(self, document_store, test_files_path):
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(content="test doc", blob=image)
        assert document_store.write_documents([doc]) == 1

        docs = document_store.filter_documents()

        assert len(docs) == 1
        assert docs[0].blob == image

    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        """
        This test has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Same reason as the filterable_docs fixture.
        Weaviate forces RFC 3339 date strings and the filterable_docs use ISO 8601 date strings.
        """
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and parser.isoparse(d.meta["date"]) > parser.isoparse("1972-12-11T19:54:58Z")
            ],
        )

    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        """
        This test has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Same reason as the filterable_docs fixture.
        Weaviate forces RFC 3339 date strings and the filterable_docs use ISO 8601 date strings.
        """
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">=", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and parser.isoparse(d.meta["date"]) >= parser.isoparse("1969-07-21T20:17:40Z")
            ],
        )

    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        """
        This test has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Same reason as the filterable_docs fixture.
        Weaviate forces RFC 3339 date strings and the filterable_docs use ISO 8601 date strings.
        """
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and parser.isoparse(d.meta["date"]) < parser.isoparse("1969-07-21T20:17:40Z")
            ],
        )

    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        """
        This test has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Same reason as the filterable_docs fixture.
        Weaviate forces RFC 3339 date strings and the filterable_docs use ISO 8601 date strings.
        """
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<=", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and parser.isoparse(d.meta["date"]) <= parser.isoparse("1969-07-21T20:17:40Z")
            ],
        )

    def test_meta_split_overlap_is_skipped(self, document_store):
        doc = Document(
            content="The moonlight shimmered ",
            meta={
                "source_id": "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0",
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
                "_split_overlap": [
                    {"doc_id": "68ed48ba830048c5d7815874ed2de794722e6d10866b6c55349a914fd9a0df65", "range": (0, 20)}
                ],
            },
        )
        document_store.write_documents([doc])

        written_doc = document_store.filter_documents()[0]

        assert written_doc.content == "The moonlight shimmered "
        assert written_doc.meta["source_id"] == "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0"
        assert written_doc.meta["page_number"] == 1.0
        assert written_doc.meta["split_id"] == 0.0
        assert written_doc.meta["split_idx_start"] == 0.0
        assert "_split_overlap" not in written_doc.meta

    def test_bm25_retrieval(self, document_store):
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
        result = document_store._bm25_retrieval("functional Haskell")
        assert len(result) == 5
        assert "functional" in result[0].content
        assert result[0].score > 0.0
        assert "functional" in result[1].content
        assert result[1].score > 0.0
        assert "functional" in result[2].content
        assert result[2].score > 0.0
        assert "functional" in result[3].content
        assert result[3].score > 0.0
        assert "functional" in result[4].content
        assert result[4].score > 0.0

    def test_bm25_retrieval_with_filters(self, document_store):
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
        filters = {"field": "content", "operator": "==", "value": "Haskell"}
        result = document_store._bm25_retrieval("functional Haskell", filters=filters)
        assert len(result) == 1
        assert "Haskell is a functional programming language" == result[0].content
        assert result[0].score > 0.0

    def test_bm25_retrieval_with_topk(self, document_store):
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
        result = document_store._bm25_retrieval("functional Haskell", top_k=3)
        assert len(result) == 3
        assert "functional" in result[0].content
        assert result[0].score > 0.0
        assert "functional" in result[1].content
        assert result[1].score > 0.0
        assert "functional" in result[2].content
        assert result[2].score > 0.0

    def test_embedding_retrieval(self, document_store):
        document_store.write_documents(
            [
                Document(
                    content="Yet another document",
                    embedding=[0.00001, 0.00001, 0.00001, 0.00002],
                ),
                Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            ]
        )
        result = document_store._embedding_retrieval(query_embedding=[1.0, 1.0, 1.0, 1.0])
        assert len(result) == 3
        assert "The document" == result[0].content
        assert result[0].score > 0.0
        assert "Another document" == result[1].content
        assert result[1].score > 0.0
        assert "Yet another document" == result[2].content
        assert result[2].score > 0.0

    def test_embedding_retrieval_with_filters(self, document_store):
        document_store.write_documents(
            [
                Document(
                    content="Yet another document",
                    embedding=[0.00001, 0.00001, 0.00001, 0.00002],
                ),
                Document(content="The document I want", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            ]
        )
        filters = {"field": "content", "operator": "==", "value": "The document I want"}
        result = document_store._embedding_retrieval(query_embedding=[1.0, 1.0, 1.0, 1.0], filters=filters)
        assert len(result) == 1
        assert "The document I want" == result[0].content
        assert result[0].score > 0.0

    def test_embedding_retrieval_with_topk(self, document_store):
        docs = [
            Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Yet another document", embedding=[0.00001, 0.00001, 0.00001, 0.00002]),
        ]
        document_store.write_documents(docs)
        results = document_store._embedding_retrieval(query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=2)
        assert len(results) == 2
        assert results[0].content == "The document"
        assert results[0].score > 0.0
        assert results[1].content == "Another document"
        assert results[1].score > 0.0

    def test_embedding_retrieval_with_distance(self, document_store):
        docs = [
            Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Yet another document", embedding=[0.00001, 0.00001, 0.00001, 0.00002]),
        ]
        document_store.write_documents(docs)
        results = document_store._embedding_retrieval(query_embedding=[1.0, 1.0, 1.0, 1.0], distance=0.0)
        assert len(results) == 1
        assert results[0].content == "The document"
        assert results[0].score > 0.0

    def test_embedding_retrieval_with_certainty(self, document_store):
        docs = [
            Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Yet another document", embedding=[0.00001, 0.00001, 0.00001, 0.00002]),
        ]
        document_store.write_documents(docs)
        results = document_store._embedding_retrieval(query_embedding=[0.8, 0.8, 0.8, 1.0], certainty=1.0)
        assert len(results) == 1
        assert results[0].content == "Another document"
        assert results[0].score > 0.0

    def test_embedding_retrieval_with_distance_and_certainty(self, document_store):
        with pytest.raises(ValueError):
            document_store._embedding_retrieval(query_embedding=[], distance=0.1, certainty=0.1)

    def test_hybrid_retrieval(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="Exilir is a functional programming language", embedding=[0.8, 0.6, 0.4, 0.3]),
                Document(content="F# is a functional programming language", embedding=[0.7, 0.5, 0.5, 0.4]),
                Document(content="C# is a functional programming language", embedding=[0.6, 0.4, 0.6, 0.5]),
                Document(content="C++ is an object oriented programming language", embedding=[0.1, 0.2, 0.8, 0.9]),
                Document(content="Dart is an object oriented programming language", embedding=[0.2, 0.3, 0.7, 0.8]),
                Document(content="Go is an object oriented programming language", embedding=[0.3, 0.4, 0.6, 0.7]),
                Document(content="Python is a object oriented programming language", embedding=[0.4, 0.5, 0.5, 0.6]),
                Document(content="Ruby is a object oriented programming language", embedding=[0.5, 0.6, 0.4, 0.5]),
                Document(content="PHP is a object oriented programming language", embedding=[0.6, 0.7, 0.3, 0.4]),
            ]
        )
        result = document_store._hybrid_retrieval("functional Haskell", query_embedding=[1.0, 0.8, 0.2, 0.1])
        assert len(result) > 0
        # Should find documents containing "functional" and similar to the embedding
        assert result[0].content == "Haskell is a functional programming language"
        assert result[0].score > 0.0

    def test_hybrid_retrieval_with_filters(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="C++ is an object oriented programming language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )
        filters = {"field": "content", "operator": "==", "value": "Haskell is a functional programming language"}
        result = document_store._hybrid_retrieval("functional", query_embedding=[1.0, 0.8, 0.2, 0.1], filters=filters)
        assert len(result) == 1
        assert result[0].content == "Haskell is a functional programming language"
        assert result[0].score > 0.0

    def test_hybrid_retrieval_with_topk(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="Exilir is a functional programming language", embedding=[0.8, 0.6, 0.4, 0.3]),
                Document(content="F# is a functional programming language", embedding=[0.7, 0.5, 0.5, 0.4]),
                Document(content="C# is a functional programming language", embedding=[0.6, 0.4, 0.6, 0.5]),
            ]
        )
        result = document_store._hybrid_retrieval("functional", query_embedding=[1.0, 0.8, 0.2, 0.1], top_k=3)
        assert len(result) == 3
        assert all("functional" in doc.content for doc in result)
        assert all(doc.score is not None and doc.score > 0.0 for doc in result)

    def test_hybrid_retrieval_with_alpha(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="C++ is an object oriented programming language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )
        # Test with alpha=0.0 (pure BM25)
        result_bm25 = document_store._hybrid_retrieval("functional", query_embedding=[1.0, 0.8, 0.2, 0.1], alpha=0.0)
        assert len(result_bm25) > 0
        assert result_bm25[0].score > 0.0

        # Test with alpha=1.0 (pure vector search)
        result_vector = document_store._hybrid_retrieval("functional", query_embedding=[1.0, 0.8, 0.2, 0.1], alpha=1.0)
        assert len(result_vector) > 0
        assert result_vector[0].score > 0.0

        # Test with alpha=0.5 (balanced hybrid)
        result_hybrid = document_store._hybrid_retrieval("functional", query_embedding=[1.0, 0.8, 0.2, 0.1], alpha=0.5)
        assert len(result_hybrid) > 0
        assert result_hybrid[0].score > 0.0

    def test_hybrid_retrieval_with_max_vector_distance(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="C++ is an object oriented programming language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )
        # Use a restrictive max_vector_distance to limit results
        result = document_store._hybrid_retrieval(
            "functional", query_embedding=[1.0, 0.8, 0.2, 0.1], max_vector_distance=0.5
        )
        assert len(result) >= 1  # Should find at least the closest match
        assert all(doc.score is not None and doc.score > 0.0 for doc in result)

    def test_hybrid_retrieval_empty_query(self, document_store):
        document_store.write_documents(
            [
                Document(content="Test document", embedding=[1.0, 0.8, 0.2, 0.1]),
            ]
        )
        # Test with empty query string
        result = document_store._hybrid_retrieval("", query_embedding=[1.0, 0.8, 0.2, 0.1])
        assert len(result) >= 0  # Should handle empty query gracefully

    def test_hybrid_retrieval_combined_parameters(self, document_store):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="Exilir is a functional programming language", embedding=[0.8, 0.6, 0.4, 0.3]),
                Document(content="C++ is an object oriented programming language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )
        # Test combining multiple parameters
        result = document_store._hybrid_retrieval(
            "functional", query_embedding=[1.0, 0.8, 0.2, 0.1], top_k=2, alpha=0.7, max_vector_distance=0.8
        )
        assert len(result) <= 2  # Should respect top_k limit
        assert all(doc.score is not None and doc.score > 0.0 for doc in result)

    def test_filter_documents_below_default_limit(self, document_store):
        docs = []
        for index in range(9998):
            docs.append(Document(content="This is some content", meta={"index": index}))
        document_store.write_documents(docs)
        result = document_store.filter_documents(
            {"field": "content", "operator": "==", "value": "This is some content"}
        )

        assert len(result) == 9998

    def test_filter_documents_over_default_limit(self, document_store):
        docs = []
        for index in range(10000):
            docs.append(Document(content="This is some content", meta={"index": index}))
        document_store.write_documents(docs)
        with pytest.raises(DocumentStoreError):
            document_store.filter_documents({"field": "content", "operator": "==", "value": "This is some content"})

    def test_schema_class_name_conversion_preserves_pascal_case(self):
        collection_settings = {"class": "CaseDocument"}
        doc_score = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        assert doc_score._collection_settings["class"] == "CaseDocument"

        collection_settings = {"class": "lower_case_name"}
        doc_score = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        assert doc_score._collection_settings["class"] == "Lower_case_name"

    @pytest.mark.skipif(
        not os.environ.get("WEAVIATE_API_KEY", None) and not os.environ.get("WEAVIATE_CLOUD_CLUSTER_URL", None),
        reason="Both WEAVIATE_API_KEY and WEAVIATE_CLOUD_CLUSTER_URL are not set. Skipping test.",
    )
    def test_connect_to_weaviate_cloud(self):
        document_store = WeaviateDocumentStore(
            url=os.environ.get("WEAVIATE_CLOUD_CLUSTER_URL"),
            auth_client_secret=AuthApiKey(api_key=Secret.from_env_var("WEAVIATE_API_KEY")),
        )
        assert document_store.client

    def test_connect_to_local(self):
        document_store = WeaviateDocumentStore(
            url="http://localhost:8080",
        )
        assert document_store.client

    def test_connect_to_embedded(self):
        document_store = WeaviateDocumentStore(embedded_options=EmbeddedOptions())
        assert document_store.client

    def test_delete_all_documents_recreate(self, document_store):
        docs = [Document(content="test doc 1"), Document(content="test doc 2")]
        assert document_store.write_documents(docs) == 2
        assert document_store.count_documents() == 2

        cls = document_store._collection_settings["class"]
        collection = document_store.client.collections.get(cls)
        previous_config = collection.config.get().to_dict()

        document_store.delete_all_documents(recreate_index=True)
        assert document_store.count_documents() == 0

        new_config = document_store.client.collections.get(cls).config.get().to_dict()
        assert previous_config == new_config

    def test_delete_all_documents_batch_size(self, document_store):
        docs = [Document(content=str(i)) for i in range(0, 5)]
        assert document_store.write_documents(docs) == 5
        document_store.delete_all_documents(batch_size=2)
        assert document_store.count_documents() == 0

    def test_delete_all_documents_excessive_batch_size(self, document_store, caplog):
        """Test that the deletion is not complete if the batch size exceeds the QUERY_MAXIMUM_RESULTS."""
        # assume QUERY_MAXIMUM_RESULTS == 10000 with standard deployment
        docs = [Document(content=str(i)) for i in range(0, 10005)]
        assert document_store.write_documents(docs) == 10005
        with caplog.at_level(logging.WARNING):
            document_store.delete_all_documents(batch_size=20000)
        assert document_store.count_documents() == 5
        assert "Not all documents have been deleted." in caplog.text

    def test_update_by_filter_with_pagination(self, document_store, monkeypatch):
        # Reduce DEFAULT_QUERY_LIMIT to test pagination without creating 10000+ documents
        monkeypatch.setattr("haystack_integrations.document_stores.weaviate.document_store.DEFAULT_QUERY_LIMIT", 100)

        docs = []
        for index in range(250):
            docs.append(
                Document(content="This is some content", meta={"index": index, "status": "draft", "category": "test"})
            )
        document_store.write_documents(docs)

        # update all documents should trigger pagination (3 pages)
        updated_count = document_store.update_by_filter(
            filters={"field": "category", "operator": "==", "value": "test"},
            meta={"status": "published"},
        )
        assert updated_count == 250

        # verify updates were correct
        published_docs = document_store.filter_documents(
            filters={"field": "status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 250
        for doc in published_docs:
            assert doc.meta["category"] == "test"
            assert doc.meta["status"] == "published"
            assert "index" in doc.meta
            assert 0 <= doc.meta["index"] < 250

    def test_count_documents_by_filter(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
            Document(content="Doc 3", meta={"category": "TypeA"}),
            Document(content="Doc 4", meta={"category": "TypeA"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count = document_store.count_documents_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "TypeA"}
        )
        assert count == 3

        count = document_store.count_documents_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "TypeB"}
        )
        assert count == 1

        count = document_store.count_documents_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "TypeC"}
        )
        assert count == 0

    def test_get_metadata_fields_info(self, document_store):
        fields_info = document_store.get_metadata_fields_info()

        # Verify special fields are excluded
        assert "_original_id" not in fields_info
        assert "content" not in fields_info
        assert "blob_data" not in fields_info
        assert "blob_mime_type" not in fields_info
        assert "score" not in fields_info

        assert "number" in fields_info
        assert fields_info["number"]["type"] == "int"
        assert "date" in fields_info
        assert fields_info["date"]["type"] == "date"
        assert "category" in fields_info
        assert fields_info["category"]["type"] == "text"
        assert "status" in fields_info
        assert fields_info["status"]["type"] == "text"

    def test_get_metadata_field_min_max(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"number": 10}),
            Document(content="Doc 2", meta={"number": 5}),
            Document(content="Doc 3", meta={"number": 20}),
            Document(content="Doc 4", meta={"number": 15}),
        ]
        document_store.write_documents(docs)

        result = document_store.get_metadata_field_min_max("number")
        assert result["min"] == 5
        assert result["max"] == 20

    def test_get_metadata_field_min_max_with_meta_prefix(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"number": 100}),
            Document(content="Doc 2", meta={"number": 200}),
        ]
        document_store.write_documents(docs)

        result = document_store.get_metadata_field_min_max("meta.number")
        assert result["min"] == 100
        assert result["max"] == 200

    def test_get_metadata_field_min_max_unsupported_type(self, document_store):
        with pytest.raises(ValueError, match="doesn't support min/max aggregation"):
            document_store.get_metadata_field_min_max("category")

    def test_get_metadata_field_min_max_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="not found in collection schema"):
            document_store.get_metadata_field_min_max("nonexistent_field")

    def test_count_unique_metadata_by_filter(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "TypeB", "status": "published"}),
            Document(content="Doc 3", meta={"category": "TypeA", "status": "draft"}),
            Document(content="Doc 4", meta={"category": "TypeC", "status": "published"}),
            Document(content="Doc 5", meta={"category": "TypeA", "status": "archived"}),
        ]
        document_store.write_documents(docs)

        result = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "TypeA"}, metadata_fields=["status"]
        )
        assert result["status"] == 2

        result = document_store.count_unique_metadata_by_filter(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "TypeA"},
                    {"field": "meta.category", "operator": "==", "value": "TypeB"},
                ],
            },
            metadata_fields=["category", "status"],
        )
        assert result["category"] == 2
        assert result["status"] == 3

    def test_count_unique_metadata_by_filter_with_meta_prefix(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
        ]
        document_store.write_documents(docs)

        result = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.category", "operator": "in", "value": ["TypeA", "TypeB"]},
            metadata_fields=["meta.category"],
        )
        assert result["category"] == 2

    def test_count_unique_metadata_by_filter_no_matches(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
        ]
        document_store.write_documents(docs)

        result = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "NonExistent"},
            metadata_fields=["category"],
        )
        assert result["category"] == 0

    def test_count_unique_metadata_by_filter_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="Fields not found in collection schema"):
            document_store.count_unique_metadata_by_filter(
                filters={"field": "meta.category", "operator": "==", "value": "TypeA"},
                metadata_fields=["nonexistent_field"],
            )

    def test_get_metadata_field_unique_values(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
            Document(content="Doc 3", meta={"category": "TypeA"}),
            Document(content="Doc 4", meta={"category": "TypeC"}),
            Document(content="Doc 5", meta={"category": "TypeB"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("category")
        assert total_count == 3
        assert set(values) == {"TypeA", "TypeB", "TypeC"}

    def test_get_metadata_field_unique_values_with_meta_prefix(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("meta.category")
        assert total_count == 2
        assert set(values) == {"TypeA", "TypeB"}

    def test_get_metadata_field_unique_values_with_search_term(self, document_store):
        docs = [
            Document(content="Python programming language", meta={"category": "TypeA"}),
            Document(content="Java programming language", meta={"category": "TypeB"}),
            Document(content="Python is great", meta={"category": "TypeC"}),
            Document(content="JavaScript tutorial", meta={"category": "TypeD"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("category", search_term="Python")
        assert total_count == 2
        assert set(values) == {"TypeA", "TypeC"}

    def test_get_metadata_field_unique_values_with_pagination(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
            Document(content="Doc 3", meta={"category": "TypeC"}),
            Document(content="Doc 4", meta={"category": "TypeD"}),
            Document(content="Doc 5", meta={"category": "TypeE"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("category", from_=0, size=2)
        assert total_count == 5
        assert len(values) == 2

        values2, total_count2 = document_store.get_metadata_field_unique_values("category", from_=2, size=2)
        assert total_count2 == 5
        assert len(values2) == 2

        assert set(values).isdisjoint(set(values2))

    def test_get_metadata_field_unique_values_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="not found in collection schema"):
            document_store.get_metadata_field_unique_values("nonexistent_field")

    def test_get_metadata_field_unique_values_empty_result(self, document_store):
        values, total_count = document_store.get_metadata_field_unique_values("category")
        assert total_count == 0
        assert values == []
