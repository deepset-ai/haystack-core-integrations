import base64
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.document import Document
from haystack.testing.document_store import CountDocumentsTest
from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)
from weaviate.auth import AuthApiKey
from weaviate.config import Config
from weaviate.embedded import (
    DEFAULT_BINARY_PATH,
    DEFAULT_GRPC_PORT,
    DEFAULT_PERSISTENCE_DATA_PATH,
    DEFAULT_PORT,
    EmbeddedOptions,
)


class TestWeaviateDocumentStore(CountDocumentsTest):
    @pytest.fixture
    def document_store(self, request) -> WeaviateDocumentStore:
        # Use a different index for each test so we can run them in parallel
        collection_settings = {"class": f"{request.node.name}"}
        store = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        yield store
        store._client.schema.delete_class(collection_settings["class"])

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate.Client")
    def test_init(self, mock_weaviate_client_class):
        mock_client = MagicMock()
        mock_client.schema.exists.return_value = False
        mock_weaviate_client_class.return_value = mock_client

        WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings={"class": "My_collection"},
            auth_client_secret=AuthApiKey("my_api_key"),
            proxies={"http": "http://proxy:1234"},
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.0",
                hostname="127.0.0.1",
            ),
            additional_config=Config(grpc_port_experimental=12345),
        )

        # Verify client is created with correct parameters
        mock_weaviate_client_class.assert_called_once_with(
            url="http://localhost:8080",
            auth_client_secret=AuthApiKey("my_api_key"),
            timeout_config=(10, 60),
            proxies={"http": "http://proxy:1234"},
            trust_env=False,
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            startup_period=5,
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.0",
                hostname="127.0.0.1",
            ),
            additional_config=Config(grpc_port_experimental=12345),
        )

        # Verify collection is created
        mock_client.schema.get.assert_called_once()
        mock_client.schema.exists.assert_called_once_with("My_collection")
        mock_client.schema.create_class.assert_called_once_with(
            {"class": "My_collection", "properties": DOCUMENT_COLLECTION_PROPERTIES}
        )

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
    def test_to_dict(self, _mock_weaviate):
        document_store = WeaviateDocumentStore(
            url="http://localhost:8080",
            auth_client_secret=AuthApiKey("my_api_key"),
            proxies={"http": "http://proxy:1234"},
            additional_headers={"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
            embedded_options=EmbeddedOptions(
                persistence_data_path=DEFAULT_PERSISTENCE_DATA_PATH,
                binary_path=DEFAULT_BINARY_PATH,
                version="1.23.0",
                hostname="127.0.0.1",
            ),
            additional_config=Config(grpc_port_experimental=12345),
        )
        assert document_store.to_dict() == {
            "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
            "init_parameters": {
                "url": "http://localhost:8080",
                "collection_settings": {
                    "class": "Default",
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
                    "type": "weaviate.auth.AuthApiKey",
                    "init_parameters": {"api_key": "my_api_key"},
                },
                "timeout_config": (10, 60),
                "proxies": {"http": "http://proxy:1234"},
                "trust_env": False,
                "additional_headers": {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
                "startup_period": 5,
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
                    "grpc_port_experimental": 12345,
                    "connection_config": {
                        "session_pool_connections": 20,
                        "session_pool_maxsize": 20,
                    },
                },
            },
        }

    @patch("haystack_integrations.document_stores.weaviate.document_store.weaviate")
    def test_from_dict(self, _mock_weaviate):
        document_store = WeaviateDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.weaviate.document_store.WeaviateDocumentStore",
                "init_parameters": {
                    "url": "http://localhost:8080",
                    "collection_settings": None,
                    "auth_client_secret": {
                        "type": "weaviate.auth.AuthApiKey",
                        "init_parameters": {"api_key": "my_api_key"},
                    },
                    "timeout_config": [10, 60],
                    "proxies": {"http": "http://proxy:1234"},
                    "trust_env": False,
                    "additional_headers": {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"},
                    "startup_period": 5,
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
                        "grpc_port_experimental": 12345,
                        "connection_config": {
                            "session_pool_connections": 20,
                            "session_pool_maxsize": 20,
                        },
                    },
                },
            }
        )

        assert document_store._url == "http://localhost:8080"
        assert document_store._collection_settings == {
            "class": "Default",
            "properties": [
                {"name": "_original_id", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "dataframe", "dataType": ["text"]},
                {"name": "blob_data", "dataType": ["blob"]},
                {"name": "blob_mime_type", "dataType": ["text"]},
                {"name": "score", "dataType": ["number"]},
            ],
        }
        assert document_store._auth_client_secret == AuthApiKey("my_api_key")
        assert document_store._timeout_config == (10, 60)
        assert document_store._proxies == {"http": "http://proxy:1234"}
        assert not document_store._trust_env
        assert document_store._additional_headers == {"X-HuggingFace-Api-Key": "MY_HUGGINGFACE_KEY"}
        assert document_store._startup_period == 5
        assert document_store._embedded_options.persistence_data_path == DEFAULT_PERSISTENCE_DATA_PATH
        assert document_store._embedded_options.binary_path == DEFAULT_BINARY_PATH
        assert document_store._embedded_options.version == "1.23.0"
        assert document_store._embedded_options.port == DEFAULT_PORT
        assert document_store._embedded_options.hostname == "127.0.0.1"
        assert document_store._embedded_options.additional_env_vars is None
        assert document_store._embedded_options.grpc_port == DEFAULT_GRPC_PORT
        assert document_store._additional_config.grpc_port_experimental == 12345
        assert document_store._additional_config.connection_config.session_pool_connections == 20
        assert document_store._additional_config.connection_config.session_pool_maxsize == 20

    def test_count_not_empty(self, document_store):
        # Skipped for the time being as we don't support writing documents
        pass

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
            "meta": {"key": "value"},
        }

    def test_to_document(self, document_store, test_files_path):
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        data = {
            "_additional": {
                "vector": [1, 2, 3],
            },
            "_original_id": "123",
            "content": "some content",
            "blob_data": base64.b64encode(image.data).decode(),
            "blob_mime_type": "image/jpeg",
            "dataframe": None,
            "score": None,
            "meta": {"key": "value"},
        }

        doc = document_store._to_document(data)
        assert doc.id == "123"
        assert doc.content == "some content"
        assert doc.blob == image
        assert doc.embedding == [1, 2, 3]
        assert doc.score is None
        assert doc.meta == {"key": "value"}
