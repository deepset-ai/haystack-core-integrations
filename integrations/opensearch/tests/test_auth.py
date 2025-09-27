# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.utils.auth import Secret
from opensearchpy import AWSV4SignerAsyncAuth, Urllib3AWSV4SignerAuth

from haystack_integrations.document_stores.opensearch.auth import AsyncAWSAuth, AWSAuth
from haystack_integrations.document_stores.opensearch.document_store import (
    DEFAULT_MAX_CHUNK_BYTES,
    OpenSearchDocumentStore,
)


class TestAWSAuth:
    @pytest.fixture(autouse=True)
    def mock_boto3_session(self):
        with patch("boto3.Session") as mock_client:
            yield mock_client

    @pytest.fixture(autouse=True)
    def set_aws_env_variables(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "some_fake_id")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "some_fake_key")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "some_fake_token")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "fake_region")
        monkeypatch.setenv("AWS_PROFILE", "some_fake_profile")

    def test_init(self, mock_boto3_session):
        aws_auth = AWSAuth()
        assert isinstance(aws_auth._urllib3_aws_v4_signer_auth, Urllib3AWSV4SignerAuth)
        mock_boto3_session.assert_called_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            profile_name="some_fake_profile",
            region_name="fake_region",
        )

    def test_to_dict(self):
        aws_auth = AWSAuth()
        res = aws_auth.to_dict()
        assert res == {
            "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "aws_service": "es",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "aws_service": "es",
            },
        }
        aws_auth = AWSAuth.from_dict(data)
        assert aws_auth.aws_access_key_id.resolve_value() == "some_fake_id"
        assert aws_auth.aws_secret_access_key.resolve_value() == "some_fake_key"
        assert aws_auth.aws_session_token.resolve_value() == "some_fake_token"
        assert aws_auth.aws_region_name.resolve_value() == "fake_region"
        assert aws_auth.aws_profile_name.resolve_value() == "some_fake_profile"
        assert aws_auth.aws_service == "es"
        assert isinstance(aws_auth._urllib3_aws_v4_signer_auth, Urllib3AWSV4SignerAuth)

    def test_from_dict_no_init_parameters(self):
        data = {"type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth"}
        aws_auth = AWSAuth.from_dict(data)
        assert aws_auth.aws_access_key_id.resolve_value() == "some_fake_id"
        assert aws_auth.aws_secret_access_key.resolve_value() == "some_fake_key"
        assert aws_auth.aws_session_token.resolve_value() == "some_fake_token"
        assert aws_auth.aws_region_name.resolve_value() == "fake_region"
        assert aws_auth.aws_profile_name.resolve_value() == "some_fake_profile"
        assert aws_auth.aws_service == "es"
        assert isinstance(aws_auth._urllib3_aws_v4_signer_auth, Urllib3AWSV4SignerAuth)

    def test_from_dict_disable_env_variables(self):
        data = {
            "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
            "init_parameters": {
                "aws_access_key_id": None,
                "aws_secret_access_key": None,
                "aws_session_token": None,
                "aws_service": "aoss",
            },
        }
        aws_auth = AWSAuth.from_dict(data)
        assert aws_auth.aws_access_key_id is None
        assert aws_auth.aws_secret_access_key is None
        assert aws_auth.aws_session_token is None
        assert aws_auth.aws_region_name.resolve_value() == "fake_region"
        assert aws_auth.aws_profile_name.resolve_value() == "some_fake_profile"
        assert aws_auth.aws_service == "aoss"
        assert isinstance(aws_auth._urllib3_aws_v4_signer_auth, Urllib3AWSV4SignerAuth)

    @patch("haystack_integrations.document_stores.opensearch.auth.AWSAuth._get_aws_v4_signer_auth")
    def test_call(self, _get_aws_v4_signer_auth):
        signer_auth_mock = Mock()
        _get_aws_v4_signer_auth.return_value = signer_auth_mock
        aws_auth = AWSAuth()
        aws_auth(method="GET", url="http://some.url", body="some body")
        signer_auth_mock.assert_called_once_with("GET", "http://some.url", "some body")

    @patch("haystack_integrations.document_stores.opensearch.auth.AWSAuth._get_aws_v4_signer_auth")
    def test_call_async(self, _get_aws_v4_signer_auth):
        signer_auth_mock = Mock()
        _get_aws_v4_signer_auth.return_value = signer_auth_mock
        async_aws_auth = AsyncAWSAuth(AWSAuth())
        async_aws_auth(method="GET", url="http://some.url", query_string="", body="some body")
        signer_auth_mock.assert_called_once_with("GET", "http://some.url", "", "some body")

    def test_async_aws_auth_init(self):
        data = {
            "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
            "init_parameters": {
                "aws_access_key_id": None,
                "aws_secret_access_key": None,
                "aws_session_token": None,
                "aws_service": "aoss",
            },
        }
        async_aws_auth = AsyncAWSAuth(AWSAuth.from_dict(data))
        assert async_aws_auth.aws_auth.aws_access_key_id is None
        assert async_aws_auth.aws_auth.aws_secret_access_key is None
        assert async_aws_auth.aws_auth.aws_session_token is None
        assert async_aws_auth.aws_auth.aws_region_name.resolve_value() == "fake_region"
        assert async_aws_auth.aws_auth.aws_profile_name.resolve_value() == "some_fake_profile"
        assert async_aws_auth.aws_auth.aws_service == "aoss"
        assert isinstance(async_aws_auth._async_aws_v4_signer_auth, AWSV4SignerAsyncAuth)


class TestDocumentStoreWithAuth:
    @pytest.fixture(autouse=True)
    def mock_boto3_session(self):
        with patch("boto3.Session") as mock_client:
            yield mock_client

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_init_with_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="testhost", http_auth=("user", "pw"))
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ("user", "pw")

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_init_without_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] is None

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_init_aws_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(
            hosts="testhost",
            http_auth=AWSAuth(aws_region_name=Secret.from_token("dummy-region")),
            use_ssl=True,
            verify_certs=True,
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert isinstance(_mock_opensearch_client.call_args[1]["http_auth"], AWSAuth)
        assert _mock_opensearch_client.call_args[1]["use_ssl"] is True
        assert _mock_opensearch_client.call_args[1]["verify_certs"] is True

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_from_dict_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {
                    "hosts": "testhost",
                    "http_auth": ["user", "pw"],
                    "use_ssl": True,
                    "verify_certs": True,
                },
            }
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pw"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_from_dict_aws_auth(self, _mock_opensearch_client, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "dummy-region")
        document_store = OpenSearchDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {
                    "hosts": "testhost",
                    "http_auth": {
                        "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
                        "init_parameters": {},
                    },
                    "use_ssl": True,
                    "verify_certs": True,
                },
            }
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert isinstance(_mock_opensearch_client.call_args[1]["http_auth"], AWSAuth)
        assert _mock_opensearch_client.call_args[1]["use_ssl"] is True
        assert _mock_opensearch_client.call_args[1]["verify_certs"] is True

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_to_dict_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="some hosts", http_auth=("user", "pw"))
        res = document_store.to_dict()
        assert res == {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "embedding_dim": 768,
                "hosts": "some hosts",
                "index": "default",
                "mappings": {
                    "dynamic_templates": [
                        {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                    ],
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                    },
                },
                "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                "method": None,
                "settings": {"index.knn": True},
                "return_embedding": False,
                "create_index": True,
                "http_auth": ("user", "pw"),
                "use_ssl": None,
                "verify_certs": None,
                "timeout": None,
            },
        }

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_to_dict_aws_auth(self, _mock_opensearch_client, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "dummy-region")
        document_store = OpenSearchDocumentStore(hosts="some hosts", http_auth=AWSAuth())
        res = document_store.to_dict()
        assert res == {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "embedding_dim": 768,
                "hosts": "some hosts",
                "index": "default",
                "mappings": {
                    "dynamic_templates": [
                        {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                    ],
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                    },
                },
                "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                "method": None,
                "settings": {"index.knn": True},
                "return_embedding": False,
                "create_index": True,
                "http_auth": {
                    "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
                    "init_parameters": {
                        "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                        "aws_secret_access_key": {
                            "type": "env_var",
                            "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                            "strict": False,
                        },
                        "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                        "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                        "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                        "aws_service": "es",
                    },
                },
                "use_ssl": None,
                "verify_certs": None,
                "timeout": None,
            },
        }

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_init_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test the default initialization using environment variables"""
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pass"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_init_with_missing_env_vars(self, _mock_opensearch_client):
        """Test that auth is None when environment variables are missing"""
        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] is None

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_to_dict_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test serialization with environment variables"""
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        document_store = OpenSearchDocumentStore(hosts="testhost")
        serialized = document_store.to_dict()

        assert "http_auth" in serialized["init_parameters"]
        auth = serialized["init_parameters"]["http_auth"]
        assert isinstance(auth, list)
        assert len(auth) == 2
        # Check that we have two Secret dictionaries with correct env vars
        assert auth[0]["type"] == "env_var"
        assert auth[0]["env_vars"] == ["OPENSEARCH_USERNAME"]
        assert auth[1]["type"] == "env_var"
        assert auth[1]["env_vars"] == ["OPENSEARCH_PASSWORD"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_ds_from_dict_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test deserialization with environment variables"""
        # Set environment variables so the secrets resolve properly
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        data = {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "hosts": "testhost",
                "http_auth": [
                    {"type": "env_var", "env_vars": ["OPENSEARCH_USERNAME"], "strict": False},
                    {"type": "env_var", "env_vars": ["OPENSEARCH_PASSWORD"], "strict": False},
                ],
            },
        }
        document_store = OpenSearchDocumentStore.from_dict(data)
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pass"]
