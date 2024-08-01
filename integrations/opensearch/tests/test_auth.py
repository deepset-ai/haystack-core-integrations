from unittest.mock import Mock, patch

import pytest
from haystack_integrations.document_stores.opensearch.auth import AWSAuth
from opensearchpy import Urllib3AWSV4SignerAuth


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

    @patch("haystack_integrations.document_stores.opensearch.auth.AWSAuth._get_urllib3_aws_v4_signer_auth")
    def test_call(self, _get_urllib3_aws_v4_signer_auth_mock):
        signer_auth_mock = Mock(spec=Urllib3AWSV4SignerAuth)
        _get_urllib3_aws_v4_signer_auth_mock.return_value = signer_auth_mock
        aws_auth = AWSAuth()
        aws_auth(method="GET", url="http://some.url", body="some body")
        signer_auth_mock.assert_called_once_with("GET", "http://some.url", "some body")
