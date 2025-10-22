from pathlib import Path
from unittest.mock import Mock

import pytest
from haystack.utils import Secret

from haystack_integrations.components.retrievers.snowflake.auth import (
    PrivateKeyReadError,
    SnowflakeAuthenticator,
)


class TestSnowflakeAuthenticator:
    """Tests for the SnowflakeAuthenticator class."""

    def test_authenticator_read_private_key_not_found(self, mocker: Mock) -> None:
        # Test error handling when private key file doesn't exist

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token("/nonexistent/key.pem"),
        )

        with pytest.raises(PrivateKeyReadError, match="Private key file not found"):
            auth.read_private_key_content()

    def test_authenticator_read_private_key_read_error(self, mocker: Mock, tmp_path: Path) -> None:
        # Test error handling when reading private key file fails

        key_file = tmp_path / "key.pem"
        key_file.write_text("test")

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
        )

        # Mock Path.read_text to raise an exception
        mocker.patch("pathlib.Path.read_text", side_effect=OSError("Permission denied"))

        with pytest.raises(PrivateKeyReadError, match="Failed to read private key file"):
            auth.read_private_key_content()

    def test_authenticator_build_jwt_params_with_fallback(self, mocker: Mock, tmp_path: Path) -> None:
        # Test JWT auth params building with fallback to file path

        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----")

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
        )

        # Mock read_private_key_content to raise an exception
        mocker.patch.object(auth, "read_private_key_content", side_effect=Exception("Read failed"))

        params = auth.build_auth_params(user="test_user")

        assert any("adbc.snowflake.sql.auth_type=auth_jwt" in p for p in params)
        assert any("username=test_user" in p for p in params)
        assert any("private_key_file=" in p for p in params)

    def test_authenticator_validate_api_key_resolution_error(self, mocker: Mock) -> None:
        # Test validation when api_key resolution fails

        mock_secret = mocker.Mock()
        mock_secret.resolve_value.side_effect = Exception("Resolution failed")

        with pytest.raises(ValueError, match="None of the following authentication environment"):
            retriever = SnowflakeAuthenticator(authenticator="SNOWFLAKE", api_key=Secret.from_env_var("TEST_ENV"))
            retriever.warm_up()

    def test_authenticator_validate_empty_api_key(self, mocker: Mock) -> None:
        # Test validation when api_key resolves to empty string

        mock_secret = mocker.Mock()
        mock_secret.resolve_value.return_value = ""  # Empty string

        with pytest.raises(ValueError, match="api_key is required"):
            SnowflakeAuthenticator(
                authenticator="SNOWFLAKE",
                api_key=mock_secret,
            )

    def test_authenticator_read_private_key_none_path(self) -> None:
        # Test read_private_key_content when private_key_file resolves to None

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token("test"),
        )

        # Mock resolve_secret_value to return None
        auth.private_key_file = None
        result = auth.read_private_key_content()
        assert result is None

    def test_authenticator_build_jwt_params_no_user(self, tmp_path: Path) -> None:
        # Test JWT params building without user parameter

        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----")

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
        )

        params = auth.build_auth_params()  # No user parameter

        assert any("adbc.snowflake.sql.auth_type=auth_jwt" in p for p in params)
        # Should not have username parameter when user is not provided
        assert not any("username=" in p for p in params)

    def test_authenticator_build_jwt_params_with_password(self, tmp_path: Path) -> None:
        # Test JWT params with encrypted private key (with password)

        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN ENCRYPTED PRIVATE KEY-----\ntest\n-----END ENCRYPTED PRIVATE KEY-----")

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
            private_key_file_pwd=Secret.from_token("test_password"),
        )

        params = auth.build_auth_params(user="test_user")

        assert any("adbc.snowflake.sql.auth_type=auth_jwt" in p for p in params)
        assert any("adbc.snowflake.sql.client_option.jwt_private_key_pkcs8_password=test_password" in p for p in params)

    def test_authenticator_build_jwt_params_empty_password(self, tmp_path: Path, mocker: Mock) -> None:
        # Test JWT params when password resolves to empty string (shouldn't be added)

        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----")

        mock_pwd_secret = mocker.Mock()
        mock_pwd_secret.resolve_value.return_value = ""  # Empty string

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
            private_key_file_pwd=mock_pwd_secret,
        )

        params = auth.build_auth_params(user="test_user")

        # Password parameter should not be included when it's an empty string
        assert not any("jwt_private_key_pkcs8_password" in p for p in params)

    def test_authenticator_build_oauth_params_with_urls(self) -> None:
        # Test OAuth params with optional URLs

        auth = SnowflakeAuthenticator(
            authenticator="OAUTH",
            oauth_client_id=Secret.from_token("client_id"),
            oauth_client_secret=Secret.from_token("client_secret"),
            oauth_token_request_url="https://token.url",
            oauth_authorization_url="https://auth.url",
        )

        params = auth.build_auth_params()

        assert any("oauth_token_request_url=https://token.url" in p for p in params)
        assert any("oauth_authorization_url=https://auth.url" in p for p in params)

    def test_authenticator_test_connection_with_database(self, mocker: Mock) -> None:
        # Test test_connection with database parameter

        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE",
            api_key=Secret.from_token("test_password"),
        )

        result = auth.test_connection(user="test_user", account="test_account", database="test_db")

        assert result is True
        mock_connect.assert_called_once()
        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["database"] == "test_db"
        mock_connection.close.assert_called_once()

    def test_authenticator_test_connection_jwt_with_pwd(self, mocker: Mock, tmp_path: Path) -> None:
        # Test test_connection for JWT with encrypted key (has password)

        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN ENCRYPTED PRIVATE KEY-----\ntest\n-----END ENCRYPTED PRIVATE KEY-----")

        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        auth = SnowflakeAuthenticator(
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_token(str(key_file)),
            private_key_file_pwd=Secret.from_token("key_password"),
        )

        result = auth.test_connection(user="test_user", account="test_account")

        assert result is True
        call_kwargs = mock_connect.call_args[1]
        assert "private_key_file_pwd" in call_kwargs
        assert call_kwargs["private_key_file_pwd"] == "key_password"

    def test_authenticator_test_connection_oauth_with_token_url(self, mocker: Mock) -> None:
        # Test test_connection for OAuth with token request URL

        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        auth = SnowflakeAuthenticator(
            authenticator="OAUTH",
            oauth_client_id=Secret.from_token("client_id"),
            oauth_client_secret=Secret.from_token("client_secret"),
            oauth_token_request_url="https://token.url",
        )

        result = auth.test_connection(user="test_user", account="test_account")

        assert result is True
        call_kwargs = mock_connect.call_args[1]
        assert "oauth_token_request_url" in call_kwargs
        assert call_kwargs["oauth_token_request_url"] == "https://token.url"
