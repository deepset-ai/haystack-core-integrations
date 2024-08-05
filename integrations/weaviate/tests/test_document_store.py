# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import random
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from dateutil import parser
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    TEST_EMBEDDING_2,
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    WriteDocumentsTest,
)
from haystack.utils.auth import Secret
from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import float32 as np_float32
from pandas import DataFrame
from weaviate.collections.classes.data import DataObject
from weaviate.config import AdditionalConfig, ConnectionConfig, Proxies, Timeout
from weaviate.embedded import (
    DEFAULT_BINARY_PATH,
    DEFAULT_GRPC_PORT,
    DEFAULT_PERSISTENCE_DATA_PATH,
    DEFAULT_PORT,
    EmbeddedOptions,
)


@patch("haystack_integrations.document_stores.weaviate.document_store.weaviate.WeaviateClient")
def test_init_is_lazy(_mock_client):
    _ = WeaviateDocumentStore()
    _mock_client.assert_not_called()


@pytest.mark.integration
class TestWeaviateDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest):
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
            ],
        }
        store = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        yield store
        store.client.collections.delete(collection_settings["class"])

    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        """
        This fixture has been copied from haystack/testing/document_store.py and modified to
        use a different date format.
        Weaviate forces RFC 3339 date strings.
        The original fixture uses ISO 8601 date strings.
        """
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40Z",
                    },
                    embedding=[random.random() for _ in range(768)],  # noqa: S311
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": -2,
                        "date": "1972-12-11T19:54:58Z",
                    },
                    embedding=[random.random() for _ in range(768)],  # noqa: S311
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00Z",
                    },
                    embedding=[random.random() for _ in range(768)],  # noqa: S311
                )
            )
            documents.append(
                Document(
                    content=f"Document {i} without embedding",
                    meta={"name": f"name_{i}", "no_embedding": True, "chapter": "conclusion"},
                )
            )
            documents.append(Document(dataframe=DataFrame([i]), meta={"name": f"table_doc_{i}"}))
            documents.append(
                Document(content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=TEST_EMBEDDING_1)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=TEST_EMBEDDING_2)
            )
        return documents

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        assert len(received) == len(expected)
        received = sorted(received, key=lambda doc: doc.id)
        expected = sorted(expected, key=lambda doc: doc.id)
        for received_doc, expected_doc in zip(received, expected):
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
                        {"name": "dataframe", "dataType": ["text"]},
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
                    },
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
                {"name": "dataframe", "dataType": ["text"]},
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

    def test_to_data_object(self, document_store, test_files_path):
        doc = Document(content="test doc")
        data = document_store._to_data_object(doc)
        assert data == {
            "_original_id": doc.id,
            "content": doc.content,
            "dataframe": None,
            "score": None,
        }

        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(
            content="test doc",
            blob=image,
            embedding=[1, 2, 3],
            meta={"key": "value"},
        )
        data = document_store._to_data_object(doc)
        assert data == {
            "_original_id": doc.id,
            "content": doc.content,
            "blob_data": base64.b64encode(image.data).decode(),
            "blob_mime_type": "image/jpeg",
            "dataframe": None,
            "score": None,
            "key": "value",
        }

    def test_to_document(self, document_store, test_files_path):
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        data = DataObject(
            properties={
                "_original_id": "123",
                "content": "some content",
                "blob_data": base64.b64encode(image.data).decode(),
                "blob_mime_type": "image/jpeg",
                "dataframe": None,
                "score": None,
                "key": "value",
            },
            vector={"default": [1, 2, 3]},
        )

        doc = document_store._to_document(data)
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

    @pytest.mark.skip(reason="Weaviate for some reason is not returning what we expect")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs):
        return super().test_comparison_not_equal_with_dataframe(document_store, filterable_docs)

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

    def test_filter_documents_with_legacy_filters(self, document_store):
        docs = []
        for index in range(10):
            docs.append(Document(content="This is some content", meta={"index": index}))
        document_store.write_documents(docs)
        result = document_store.filter_documents({"content": {"$eq": "This is some content"}})

        assert len(result) == 10

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
