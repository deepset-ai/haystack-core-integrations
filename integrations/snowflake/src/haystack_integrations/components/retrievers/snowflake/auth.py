# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

from haystack import logging
from haystack.utils import Secret

import snowflake.connector  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# Authentication type constants
AUTH_SNOWFLAKE = "SNOWFLAKE"
AUTH_SNOWFLAKE_JWT = "SNOWFLAKE_JWT"
AUTH_OAUTH = "OAUTH"

# ADBC-specific parameters
ADBC_AUTH_TYPE_JWT = "auth_jwt"
ADBC_PARAM_AUTH_TYPE = "adbc.snowflake.sql.auth_type"
ADBC_PARAM_JWT_KEY_VALUE = "adbc.snowflake.sql.client_option.jwt_private_key_pkcs8_value"
ADBC_PARAM_JWT_KEY_PASSWORD = "adbc.snowflake.sql.client_option.jwt_private_key_pkcs8_password"
ADBC_PARAM_USERNAME = "username"

# Error messages
ERROR_PRIVATE_KEY_FILE_REQUIRED = "private_key_file is required for SNOWFLAKE_JWT authentication"
ERROR_OAUTH_CLIENT_ID_REQUIRED = "oauth_client_id is required for OAUTH authentication"
ERROR_OAUTH_CLIENT_SECRET_REQUIRED = "oauth_client_secret is required for OAUTH authentication"
ERROR_API_KEY_REQUIRED = "api_key is required for SNOWFLAKE (password) authentication"


class PrivateKeyReadError(Exception):
    """Raised when private key file cannot be read properly."""


class SnowflakeAuthenticator:
    """
    Handles Snowflake authentication for different authentication methods.

    Supports:
    - SNOWFLAKE: Standard password authentication
    - SNOWFLAKE_JWT: Key-pair JWT authentication
    - OAUTH: OAuth 2.0 authentication
    """

    def __init__(
        self,
        authenticator: Literal["SNOWFLAKE", "SNOWFLAKE_JWT", "OAUTH"],
        api_key: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_API_KEY", strict=False),  # noqa: B008
        private_key_file: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE", strict=False),  # noqa: B008
        private_key_file_pwd: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE_PWD", strict=False),  # noqa: B008
        oauth_client_id: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_CLIENT_ID", strict=False),  # noqa: B008
        oauth_client_secret: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_CLIENT_SECRET", strict=False),  # noqa: B008
        oauth_token_request_url: Optional[str] = None,
        oauth_authorization_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the authenticator with the specified authentication method.

        :param authenticator: Authentication method to use.
        :param api_key: Password for SNOWFLAKE authentication.
        :param private_key_file: Path to private key file for SNOWFLAKE_JWT authentication.
        :param private_key_file_pwd: Passphrase for private key file.
        :param oauth_client_id: OAuth client ID for OAUTH authentication.
        :param oauth_client_secret: OAuth client secret for OAUTH authentication.
        :param oauth_token_request_url: OAuth token request URL.
        :param oauth_authorization_url: OAuth authorization URL.
        """
        self.authenticator = authenticator
        self.api_key = api_key.resolve_value() if api_key else None
        self.private_key_file = private_key_file.resolve_value() if private_key_file else None
        self.private_key_file_pwd = private_key_file_pwd.resolve_value() if private_key_file_pwd else None
        self.oauth_client_id = oauth_client_id.resolve_value() if oauth_client_id else None
        self.oauth_client_secret = oauth_client_secret.resolve_value() if oauth_client_secret else None
        self.oauth_token_request_url = oauth_token_request_url
        self.oauth_authorization_url = oauth_authorization_url

        self.validate_auth_params()

    def validate_auth_params(self) -> None:
        """
        Validates authentication parameters based on the chosen authentication method.

        :raises ValueError: If required parameters are missing for the selected authentication method.
        """
        if self.authenticator == AUTH_SNOWFLAKE_JWT:
            if not self.private_key_file:
                raise ValueError(ERROR_PRIVATE_KEY_FILE_REQUIRED)
        elif self.authenticator == AUTH_OAUTH:
            if not self.oauth_client_id:
                raise ValueError(ERROR_OAUTH_CLIENT_ID_REQUIRED)
            if not self.oauth_client_secret:
                raise ValueError(ERROR_OAUTH_CLIENT_SECRET_REQUIRED)
        elif self.authenticator == AUTH_SNOWFLAKE:
            if not self.api_key:
                raise ValueError(ERROR_API_KEY_REQUIRED)

    def read_private_key_content(self) -> Optional[str]:
        """
        Reads the private key file content for ADBC compatibility.

        :returns: Private key content as a string, or None if not available.
        :raises PrivateKeyReadError: If the file cannot be read.
        """
        if not self.private_key_file:
            return None

        try:
            key_path = Path(self.private_key_file)
            if not key_path.exists():
                msg = f"Private key file not found: {self.private_key_file}"
                raise PrivateKeyReadError(msg)

            return key_path.read_text()
        except PrivateKeyReadError:
            raise
        except Exception as e:
            msg = f"Failed to read private key file: {e!s}"
            raise PrivateKeyReadError(msg) from e

    def _build_jwt_auth_params(self, user: Optional[str] = None) -> list[str]:
        """
        Builds JWT authentication parameters for ADBC.

        :param user: Username for connection.
        :returns: List of JWT authentication parameters.
        """
        params = [f"{ADBC_PARAM_AUTH_TYPE}={ADBC_AUTH_TYPE_JWT}"]

        # Add username as parameter for ADBC (since it's not in the URI for JWT)
        if user:
            params.append(f"{ADBC_PARAM_USERNAME}={user}")

        # Read private key content for ADBC
        if self.private_key_file:
            try:
                private_key_content = self.read_private_key_content()
                if private_key_content:
                    # URL encode the key content to handle newlines and special characters
                    encoded_key = quote(private_key_content, safe="")
                    params.append(f"{ADBC_PARAM_JWT_KEY_VALUE}={encoded_key}")
            except Exception as e:
                logger.warning(f"Failed to read private key content, falling back to file path: {e!s}")
                # Fallback to file path (though ADBC may not support this)
                params.append(f"private_key_file={self.private_key_file}")

        # Only include password parameter if it's actually set
        if self.private_key_file_pwd:  # Only add if not empty string
            params.append(f"{ADBC_PARAM_JWT_KEY_PASSWORD}={self.private_key_file_pwd}")

        return params

    def _build_oauth_auth_params(self) -> list[str]:
        """
        Builds OAuth authentication parameters.

        :returns: List of OAuth authentication parameters.
        """
        params = [f"authenticator={self.authenticator}"]

        if self.oauth_client_id:
            params.append(f"oauth_client_id={self.oauth_client_id}")
        if self.oauth_client_secret:
            params.append(f"oauth_client_secret={self.oauth_client_secret}")
        if self.oauth_token_request_url:
            params.append(f"oauth_token_request_url={self.oauth_token_request_url}")
        if self.oauth_authorization_url:
            params.append(f"oauth_authorization_url={self.oauth_authorization_url}")

        return params

    def build_auth_params(self, user: Optional[str] = None) -> list[str]:
        """
        Builds authentication parameters for the connection URI.

        :param user: Username for connection (required for JWT with ADBC).
        :returns: List of authentication parameters.
        :raises ValueError: If secret resolution fails.
        """
        if self.authenticator == AUTH_SNOWFLAKE_JWT:
            return self._build_jwt_auth_params(user)
        elif self.authenticator == AUTH_OAUTH:
            return self._build_oauth_auth_params()
        return []

    def get_password_for_uri(self) -> Optional[str]:
        """
        Gets the password for URI construction in SNOWFLAKE authentication.

        :returns: Resolved password value or None.
        :raises ValueError: If secret resolution fails.
        """
        if self.authenticator == AUTH_SNOWFLAKE and self.api_key:
            return self.api_key
        return None

    def create_masked_params(self, params: list) -> list[str]:
        """
        Creates a masked version of authentication parameters for safe logging.

        :param params: Original parameter list.
        :returns: Parameter list with sensitive information masked.
        """
        masked_params = []

        for param in params:
            masked_param = param

            # Mask private key password
            if self.private_key_file_pwd and self.private_key_file_pwd in param:
                masked_param = param.replace(self.private_key_file_pwd, "***REDACTED***")

            # Mask OAuth client secret
            if self.oauth_client_secret and self.oauth_client_secret in param:
                masked_param = masked_param.replace(self.oauth_client_secret, "***REDACTED***")

            masked_params.append(masked_param)

        return masked_params

    def test_connection(self, user: str, account: str, database: Optional[str] = None) -> bool:
        """
        Tests the connection with the provided credentials.

        :param user: Snowflake username.
        :param account: Snowflake account identifier.
        :param database: Optional database name.
        :returns: True if connection is successful, False otherwise.
        """
        try:
            connection_params: dict[str, Any] = {
                "user": user,
                "account": account,
                "authenticator": self.authenticator.lower(),
            }

            if database:
                connection_params["database"] = database

            if self.authenticator == AUTH_SNOWFLAKE:
                if self.api_key:
                    connection_params["password"] = self.api_key
            elif self.authenticator == AUTH_SNOWFLAKE_JWT:
                if self.private_key_file:
                    connection_params["private_key_file"] = self.private_key_file
                if self.private_key_file_pwd:
                    connection_params["private_key_file_pwd"] = self.private_key_file_pwd
            elif self.authenticator == AUTH_OAUTH:
                if self.oauth_client_id and self.oauth_client_secret:
                    connection_params["oauth_client_id"] = self.oauth_client_id
                    connection_params["oauth_client_secret"] = self.oauth_client_secret
                if self.oauth_token_request_url:
                    connection_params["oauth_token_request_url"] = self.oauth_token_request_url

            # Test connection
            conn = snowflake.connector.connect(**connection_params)
            conn.close()
            logger.info("Authentication test successful")
            return True

        except Exception as e:
            logger.warning(f"Authentication test failed: {e!s}")
            return False
