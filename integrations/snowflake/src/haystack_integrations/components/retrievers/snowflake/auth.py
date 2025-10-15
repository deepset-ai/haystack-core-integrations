# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional
from urllib.parse import quote

from haystack import logging
from haystack.utils import Secret

if TYPE_CHECKING:
    import snowflake.connector  # type: ignore[import-not-found]
else:
    try:
        import snowflake.connector  # type: ignore[import-not-found]
    except ImportError:
        snowflake = None  # type: ignore[misc]

logger = logging.getLogger(__name__)


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
        api_key: Optional[Secret] = None,
        private_key_file: Optional[Secret] = None,
        private_key_file_pwd: Optional[Secret] = None,
        oauth_client_id: Optional[Secret] = None,
        oauth_client_secret: Optional[Secret] = None,
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
        self.api_key = api_key
        self.private_key_file = private_key_file
        self.private_key_file_pwd = private_key_file_pwd
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self.oauth_token_request_url = oauth_token_request_url
        self.oauth_authorization_url = oauth_authorization_url

        self.validate_auth_params()

    def validate_auth_params(self) -> None:
        """
        Validates authentication parameters based on the chosen authentication method.

        :raises ValueError: If required parameters are missing for the selected authentication method.
        """
        if self.authenticator == "SNOWFLAKE_JWT":
            if not self.private_key_file:
                msg = "private_key_file is required for SNOWFLAKE_JWT authentication"
                raise ValueError(msg)
        elif self.authenticator == "OAUTH":
            if not self.oauth_client_id:
                msg = "oauth_client_id is required for OAUTH authentication"
                raise ValueError(msg)
            if not self.oauth_client_secret:
                msg = "oauth_client_secret is required for OAUTH authentication"
                raise ValueError(msg)
        elif self.authenticator == "SNOWFLAKE":
            if not self.api_key:
                msg = "api_key is required for SNOWFLAKE (password) authentication"
                raise ValueError(msg)
            try:
                api_key_value = self.api_key.resolve_value()
                if not api_key_value:
                    msg = "api_key is required for SNOWFLAKE (password) authentication"
                    raise ValueError(msg)
            except Exception as e:
                msg = f"Failed to resolve api_key: {e!s}"
                raise ValueError(msg) from e

    def resolve_secret_value(self, value: Optional[Secret]) -> Optional[str]:
        """
        Safely resolves a Secret value.

        :param value: Secret to resolve.
        :returns: Resolved string value or None.
        :raises ValueError: If secret resolution fails.
        """
        if value is None:
            return None
        try:
            return value.resolve_value()
        except Exception as e:
            msg = f"Failed to resolve secret value: {e!s}"
            raise ValueError(msg) from e

    def read_private_key_content(self) -> Optional[str]:
        """
        Reads the private key file content for ADBC compatibility.

        :returns: Private key content as a string, or None if not available.
        :raises ValueError: If the file cannot be read.
        """
        if not self.private_key_file:
            return None

        try:
            private_key_path = self.resolve_secret_value(self.private_key_file)
            if not private_key_path:
                return None

            key_path = Path(private_key_path)
            if not key_path.exists():
                msg = f"Private key file not found: {private_key_path}"
                raise ValueError(msg)

            return key_path.read_text()
        except Exception as e:
            msg = f"Failed to read private key file: {e!s}"
            raise ValueError(msg) from e

    def build_auth_params(self, user: Optional[str] = None) -> list[str]:
        """
        Builds authentication parameters for the connection URI.
        Uses ADBC-specific parameter format for JWT authentication.

        :param user: Username for connection (required for JWT with ADBC).
        :returns: List of authentication parameters.
        :raises ValueError: If secret resolution fails.
        """
        params = []

        if self.authenticator == "SNOWFLAKE_JWT":
            # Use ADBC-specific parameters for JWT authentication
            params.append("adbc.snowflake.sql.auth_type=auth_jwt")

            # Add username as parameter for ADBC (since it's not in the URI for JWT)
            if user:
                params.append(f"username={user}")

            # Read private key content for ADBC
            if self.private_key_file:
                try:
                    private_key_content = self.read_private_key_content()
                    if private_key_content:
                        # URL encode the key content to handle newlines and special characters
                        encoded_key = quote(private_key_content, safe="")
                        params.append(f"adbc.snowflake.sql.client_option.jwt_private_key_pkcs8_value={encoded_key}")
                except Exception as e:
                    logger.warning(f"Failed to read private key content, falling back to file path: {e!s}")
                    # Fallback to file path (though ADBC may not support this)
                    private_key_path = self.resolve_secret_value(self.private_key_file)
                    params.append(f"private_key_file={private_key_path}")

            # Only include password parameter if it's actually set
            if self.private_key_file_pwd:
                private_key_pwd = self.resolve_secret_value(self.private_key_file_pwd)
                if private_key_pwd:  # Only add if not empty string
                    params.append(f"adbc.snowflake.sql.client_option.jwt_private_key_pkcs8_password={private_key_pwd}")
        elif self.authenticator == "OAUTH":
            params.append(f"authenticator={self.authenticator}")
            if self.oauth_client_id:
                client_id = self.resolve_secret_value(self.oauth_client_id)
                params.append(f"oauth_client_id={client_id}")
            if self.oauth_client_secret:
                client_secret = self.resolve_secret_value(self.oauth_client_secret)
                params.append(f"oauth_client_secret={client_secret}")
            if self.oauth_token_request_url:
                params.append(f"oauth_token_request_url={self.oauth_token_request_url}")
            if self.oauth_authorization_url:
                params.append(f"oauth_authorization_url={self.oauth_authorization_url}")

        return params

    def get_password_for_uri(self) -> Optional[str]:
        """
        Gets the password for URI construction in SNOWFLAKE authentication.

        :returns: Resolved password value or None.
        :raises ValueError: If secret resolution fails.
        """
        if self.authenticator == "SNOWFLAKE" and self.api_key:
            return self.resolve_secret_value(self.api_key)
        return None

    def create_masked_params(self, params: list) -> list:
        """
        Creates a masked version of authentication parameters for safe logging.

        :param params: Original parameter list.
        :returns: Parameter list with sensitive information masked.
        """
        masked_params = []

        for param in params:
            masked_param = param

            # Mask private key password
            if self.private_key_file_pwd:
                private_key_pwd = self.resolve_secret_value(self.private_key_file_pwd)
                if private_key_pwd and private_key_pwd in param:
                    masked_param = param.replace(private_key_pwd, "***REDACTED***")

            # Mask OAuth client secret
            if self.oauth_client_secret:
                client_secret = self.resolve_secret_value(self.oauth_client_secret)
                if client_secret and client_secret in param:
                    masked_param = masked_param.replace(client_secret, "***REDACTED***")

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
            try:
                import snowflake.connector  # type: ignore[import-not-found]  # noqa: PLC0415
            except ImportError as e:
                msg = "snowflake-connector-python is required for connection testing"
                raise ImportError(msg) from e

            connection_params: dict[str, Any] = {
                "user": user,
                "account": account,
                "authenticator": self.authenticator.lower(),
            }

            if database:
                connection_params["database"] = database

            if self.authenticator == "SNOWFLAKE":
                password = self.resolve_secret_value(self.api_key)
                if password:
                    connection_params["password"] = password
            elif self.authenticator == "SNOWFLAKE_JWT":
                private_key_file = self.resolve_secret_value(self.private_key_file)
                if private_key_file:
                    connection_params["private_key_file"] = private_key_file
                if self.private_key_file_pwd:
                    private_key_pwd = self.resolve_secret_value(self.private_key_file_pwd)
                    if private_key_pwd:
                        connection_params["private_key_file_pwd"] = private_key_pwd
            elif self.authenticator == "OAUTH":
                client_id = self.resolve_secret_value(self.oauth_client_id)
                client_secret = self.resolve_secret_value(self.oauth_client_secret)
                if client_id and client_secret:
                    connection_params["oauth_client_id"] = client_id
                    connection_params["oauth_client_secret"] = client_secret
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
