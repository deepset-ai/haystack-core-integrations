from unittest.mock import patch

from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from weaviate.auth import AuthApiKey
from weaviate.config import Config
from weaviate.embedded import (
    DEFAULT_BINARY_PATH,
    DEFAULT_GRPC_PORT,
    DEFAULT_PERSISTENCE_DATA_PATH,
    DEFAULT_PORT,
    EmbeddedOptions,
)


class TestWeaviateDocumentStore:
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
